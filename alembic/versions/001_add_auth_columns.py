"""Add auth columns to users table

Revision ID: 001_add_auth_columns
Revises: 
Create Date: 2025-12-24

Adds authentication-related columns to the users table:
- password_hash: bcrypt hashed password
- session_token: UUID session token for API auth
- display_name: Optional display name
- last_login: Last login timestamp
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_add_auth_columns'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add auth columns to users table."""
    
    # Add password_hash column
    op.add_column(
        'users',
        sa.Column(
            'password_hash',
            sa.String(255),
            nullable=True,
            comment='bcrypt hashed password'
        )
    )
    
    # Add session_token column with index
    op.add_column(
        'users',
        sa.Column(
            'session_token',
            sa.String(64),
            nullable=True,
            comment='UUID session token for API authentication'
        )
    )
    
    # Add display_name column
    op.add_column(
        'users',
        sa.Column(
            'display_name',
            sa.String(100),
            nullable=True,
            comment='User display name'
        )
    )
    
    # Add last_login column
    op.add_column(
        'users',
        sa.Column(
            'last_login',
            sa.TIMESTAMP(),
            nullable=True,
            comment='Last login timestamp'
        )
    )
    
    # Create index on session_token for fast lookups
    op.create_index(
        'idx_users_session_token',
        'users',
        ['session_token'],
        unique=False
    )


def downgrade() -> None:
    """Remove auth columns from users table."""
    
    # Drop index first
    op.drop_index('idx_users_session_token', table_name='users')
    
    # Drop columns
    op.drop_column('users', 'last_login')
    op.drop_column('users', 'display_name')
    op.drop_column('users', 'session_token')
    op.drop_column('users', 'password_hash')
