from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv('ACCOUNT_DB_PATH', str(ROOT / 'data' / 'accounts.sqlite3')))
COOKIE_NAME = 'metals_session'
COOKIE_SECURE = os.getenv('COOKIE_SECURE', 'false').lower() == 'true'
SESSION_DAYS = 14
PASSWORD_ITERATIONS = 600_000
router = APIRouter(prefix='/api')


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat(timespec='seconds')


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys = ON')
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize_accounts() -> None:
    with connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            display_name TEXT NOT NULL,
            password_salt BLOB NOT NULL,
            password_hash BLOB NOT NULL,
            role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('admin','user')),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions(user_id);
        CREATE TABLE IF NOT EXISTS saved_trades (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            metal TEXT NOT NULL CHECK(metal IN ('gold','silver')),
            side TEXT NOT NULL CHECK(side IN ('buy','sell')),
            grams REAL NOT NULL CHECK(grams > 0),
            price_inr_per_gram REAL NOT NULL CHECK(price_inr_per_gram > 0),
            trade_date TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS trades_user_idx ON saved_trades(user_id, trade_date DESC);
        CREATE TABLE IF NOT EXISTS saved_analyses (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS analyses_user_idx ON saved_analyses(user_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS saved_uploads (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            worksheet TEXT,
            row_count INTEGER NOT NULL,
            column_count INTEGER NOT NULL,
            duplicate_count INTEGER NOT NULL,
            file_bytes BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS uploads_user_idx ON saved_uploads(user_id, created_at DESC);
        ''')


initialize_accounts()


class Credentials(BaseModel):
    email: str = Field(min_length=6, max_length=254)
    password: str = Field(min_length=10, max_length=128)
    display_name: str = Field(default='', max_length=80)


class TradeInput(BaseModel):
    metal: str
    side: str
    grams: float = Field(gt=0, le=1_000_000)
    price_inr_per_gram: float = Field(gt=0, le=1_000_000_000)
    trade_date: str = Field(min_length=10, max_length=10)
    notes: str = Field(default='', max_length=500)


class AnalysisInput(BaseModel):
    kind: str = Field(min_length=1, max_length=60)
    title: str = Field(min_length=1, max_length=200)
    payload: dict[str, Any]


def normalize_email(email: str) -> str:
    value = email.strip().lower()
    if not re.fullmatch(r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9.-]+\.[a-z]{2,}", value):
        raise HTTPException(422, 'Enter a valid email address.')
    return value


def password_digest(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, PASSWORD_ITERATIONS, dklen=32)


def public_user(row: sqlite3.Row) -> dict[str, Any]:
    return {'id': row['id'], 'email': row['email'], 'display_name': row['display_name'], 'role': row['role'], 'created_at': row['created_at']}


def start_session(user_id: int, response: Response) -> None:
    token = secrets.token_urlsafe(40)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    created = utc_now()
    expires = created + timedelta(days=SESSION_DAYS)
    with connect() as db:
        db.execute('INSERT INTO sessions(token_hash,user_id,expires_at,created_at) VALUES(?,?,?,?)', (token_hash, user_id, utc_text(expires), utc_text(created)))
        db.execute('DELETE FROM sessions WHERE expires_at < ?', (utc_text(),))
    response.set_cookie(COOKIE_NAME, token, max_age=SESSION_DAYS * 86400, httponly=True, secure=COOKIE_SECURE, samesite='lax', path='/api')


def current_user(request: Request) -> dict[str, Any]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(401, 'Sign in to access your account.')
    digest = hashlib.sha256(token.encode()).hexdigest()
    now = utc_text()
    with connect() as db:
        row = db.execute('''SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id
                            WHERE s.token_hash=? AND s.expires_at>?''', (digest, now)).fetchone()
        if row is None:
            db.execute('DELETE FROM sessions WHERE token_hash=? OR expires_at<=?', (digest, now))
            raise HTTPException(401, 'Your session expired. Please sign in again.')
        return public_user(row)


def admin_user(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if user['role'] != 'admin':
        raise HTTPException(403, 'Administrator access is required.')
    return user


@router.post('/auth/register')
def register(payload: Credentials, response: Response):
    email = normalize_email(payload.email)
    if len(payload.password) < 10:
        raise HTTPException(422, 'Use at least 10 characters for your app password.')
    name = payload.display_name.strip()[:80] or email.split('@', 1)[0]
    salt = secrets.token_bytes(16)
    digest = password_digest(payload.password, salt)
    try:
        with connect() as db:
            db.execute('BEGIN IMMEDIATE')
            first_account = db.execute('SELECT COUNT(*) AS n FROM users').fetchone()['n'] == 0
            role = 'admin' if first_account else 'user'
            result = db.execute('''INSERT INTO users(email,display_name,password_salt,password_hash,role,created_at)
                                   VALUES(?,?,?,?,?,?)''', (email, name, salt, digest, role, utc_text()))
            user_id = int(result.lastrowid)
            row = db.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
    except sqlite3.IntegrityError:
        raise HTTPException(409, 'An account with this email already exists. Sign in instead.')
    start_session(user_id, response)
    return {'user': public_user(row)}


@router.post('/auth/login')
def login(payload: Credentials, response: Response):
    email = normalize_email(payload.email)
    with connect() as db:
        row = db.execute('SELECT * FROM users WHERE email=? COLLATE NOCASE', (email,)).fetchone()
    salt = bytes(row['password_salt']) if row else b'\x00' * 16
    expected = bytes(row['password_hash']) if row else b'\x00' * 32
    matches = hmac.compare_digest(password_digest(payload.password, salt), expected)
    if row is None or not matches:
        raise HTTPException(401, 'Email or password is incorrect.')
    start_session(int(row['id']), response)
    return {'user': public_user(row)}


@router.post('/auth/logout')
def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        digest = hashlib.sha256(token.encode()).hexdigest()
        with connect() as db:
            db.execute('DELETE FROM sessions WHERE token_hash=?', (digest,))
    response.delete_cookie(COOKIE_NAME, path='/api', httponly=True, secure=COOKIE_SECURE, samesite='lax')
    return {'ok': True}


@router.get('/auth/me')
def me(user: dict[str, Any] = Depends(current_user)):
    return {'user': user}


@router.get('/account/trades')
def get_trades(user: dict[str, Any] = Depends(current_user)):
    with connect() as db:
        rows = db.execute('''SELECT id,metal,side,grams,price_inr_per_gram,trade_date,notes,created_at
                             FROM saved_trades WHERE user_id=? ORDER BY trade_date DESC,id DESC''', (user['id'],)).fetchall()
    return {'trades': [dict(row) for row in rows]}


@router.post('/account/trades')
def save_trade(payload: TradeInput, user: dict[str, Any] = Depends(current_user)):
    metal, side = payload.metal.lower(), payload.side.lower()
    if metal not in {'gold', 'silver'} or side not in {'buy', 'sell'}:
        raise HTTPException(422, 'Choose gold or silver and buy or sell.')
    try:
        trade_date = datetime.strptime(payload.trade_date, '%Y-%m-%d').date().isoformat()
    except ValueError:
        raise HTTPException(422, 'Choose a valid trade date.')
    with connect() as db:
        cur = db.execute('''INSERT INTO saved_trades(user_id,metal,side,grams,price_inr_per_gram,trade_date,notes,created_at)
                            VALUES(?,?,?,?,?,?,?,?)''', (user['id'], metal, side, payload.grams, payload.price_inr_per_gram, trade_date, payload.notes.strip(), utc_text()))
        row = db.execute('''SELECT id,metal,side,grams,price_inr_per_gram,trade_date,notes,created_at
                            FROM saved_trades WHERE id=? AND user_id=?''', (cur.lastrowid, user['id'])).fetchone()
    return {'trade': dict(row)}


@router.get('/account/analyses')
def get_analyses(user: dict[str, Any] = Depends(current_user)):
    with connect() as db:
        rows = db.execute('''SELECT id,kind,title,payload_json,created_at FROM saved_analyses
                             WHERE user_id=? ORDER BY id DESC LIMIT 100''', (user['id'],)).fetchall()
    return {'analyses': [{'id': row['id'], 'kind': row['kind'], 'title': row['title'], 'payload': json.loads(row['payload_json']), 'created_at': row['created_at']} for row in rows]}


@router.post('/account/analyses')
def save_analysis(payload: AnalysisInput, user: dict[str, Any] = Depends(current_user)):
    try:
        packed = json.dumps(payload.payload, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    except (ValueError, TypeError):
        raise HTTPException(422, 'Analysis contains a value that cannot be saved.')
    if len(packed.encode('utf-8')) > 300_000:
        raise HTTPException(413, 'This analysis is too large to save.')
    with connect() as db:
        cur = db.execute('INSERT INTO saved_analyses(user_id,kind,title,payload_json,created_at) VALUES(?,?,?,?,?)', (user['id'], payload.kind, payload.title, packed, utc_text()))
    return {'id': cur.lastrowid, 'saved': True}


def save_upload(user_id: int, filename: str, worksheet: str | None, rows: int, columns: int, duplicates: int, content: bytes) -> int:
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, 'Files must be 15 MB or smaller.')
    with connect() as db:
        cur = db.execute('''INSERT INTO saved_uploads(user_id,filename,worksheet,row_count,column_count,duplicate_count,file_bytes,created_at)
                            VALUES(?,?,?,?,?,?,?,?)''', (user_id, Path(filename).name[:255], worksheet, rows, columns, duplicates, content, utc_text()))
        return int(cur.lastrowid)


def list_uploads(user_id: int) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute('''SELECT id,filename,worksheet,row_count,column_count,duplicate_count,created_at
                             FROM saved_uploads WHERE user_id=? ORDER BY id DESC LIMIT 100''', (user_id,)).fetchall()
    return [dict(row) for row in rows]


@router.get('/account/uploads')
def get_uploads(user: dict[str, Any] = Depends(current_user)):
    return {'uploads': list_uploads(int(user['id']))}


@router.get('/admin/users')
def list_users(_: dict[str, Any] = Depends(admin_user)):
    with connect() as db:
        rows = db.execute('''SELECT u.id,u.email,u.display_name,u.role,u.created_at,
                             (SELECT COUNT(*) FROM saved_trades t WHERE t.user_id=u.id) AS trades,
                             (SELECT COUNT(*) FROM saved_analyses a WHERE a.user_id=u.id) AS analyses,
                             (SELECT COUNT(*) FROM saved_uploads f WHERE f.user_id=u.id) AS uploads
                             FROM users u ORDER BY u.id''').fetchall()
    return {'users': [dict(row) for row in rows]}


@router.get('/admin/users/{user_id}/data')
def get_user_data(user_id: int, _: dict[str, Any] = Depends(admin_user)):
    """Allow the project administrator to review saved user submissions, never credentials."""
    with connect() as db:
        user = db.execute('SELECT id,email,display_name,role,created_at FROM users WHERE id=?', (user_id,)).fetchone()
        if user is None:
            raise HTTPException(404, 'Account not found.')
        trades = db.execute('''SELECT id,metal,side,grams,price_inr_per_gram,trade_date,notes,created_at
                               FROM saved_trades WHERE user_id=? ORDER BY trade_date DESC,id DESC''', (user_id,)).fetchall()
        analyses = db.execute('''SELECT id,kind,title,payload_json,created_at FROM saved_analyses
                                 WHERE user_id=? ORDER BY id DESC LIMIT 100''', (user_id,)).fetchall()
        uploads = db.execute('''SELECT id,filename,worksheet,row_count,column_count,duplicate_count,created_at
                                FROM saved_uploads WHERE user_id=? ORDER BY id DESC LIMIT 100''', (user_id,)).fetchall()
    return {
        'user': dict(user),
        'trades': [dict(row) for row in trades],
        'analyses': [{'id': row['id'], 'kind': row['kind'], 'title': row['title'], 'payload': json.loads(row['payload_json']), 'created_at': row['created_at']} for row in analyses],
        'uploads': [dict(row) for row in uploads],
    }


def store_upload(user_id: int, filename: str, worksheet: str | None, rows: int, columns: int, duplicates: int, content: bytes) -> int:
    return save_upload(user_id, filename, worksheet, rows, columns, duplicates, content)
