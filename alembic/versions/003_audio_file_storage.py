"""Replace audio base64 with file path storage.

Revision ID: 003_audio_file_storage
Revises: 002_add_audio_base64
Create Date: 2025-12-26

Switches from base64-in-database to Opus file storage.
Audio files are stored in backend/audio_storage/{user_id}/{conversation_id}.opus
Database stores only the file path (~500 bytes vs ~1MB).
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003_audio_file_storage'
down_revision = '002_add_audio_base64'
branch_labels = None
depends_on = None


def upgrade():
    """
    Switch from base64 audio storage to file path storage.
    
    Changes:
    - Add response_audio_path (VARCHAR 500) for file path (if not exists)
    - Keep response_audio_duration_seconds (already exists)
    - Drop response_audio_base64 (TEXT blob - inefficient)
    - Drop audio_is_compressed (no longer needed)
    """
    from sqlalchemy import inspect
    from alembic import op
    
    # Get connection to check existing columns
    conn = op.get_bind()
    inspector = inspect(conn)
    existing_columns = [col['name'] for col in inspector.get_columns('conversations')]
    
    # Add new column for file path (only if it doesn't exist)
    if 'response_audio_path' not in existing_columns:
        op.add_column(
            'conversations',
            sa.Column('response_audio_path', sa.String(500), nullable=True)
        )
        print("✅ Added response_audio_path column")
    else:
        print("⏭️ response_audio_path column already exists, skipping")
    
    # Drop old columns (base64 storage is deprecated)
    if 'response_audio_base64' in existing_columns:
        op.drop_column('conversations', 'response_audio_base64')
        print("✅ Dropped response_audio_base64 column")
    else:
        print("⏭️ response_audio_base64 column doesn't exist, skipping")
    
    if 'audio_is_compressed' in existing_columns:
        op.drop_column('conversations', 'audio_is_compressed')
        print("✅ Dropped audio_is_compressed column")
    else:
        print("⏭️ audio_is_compressed column doesn't exist, skipping")
    
    print("✅ Migration complete: Audio now stored as Opus files")


def downgrade():
    """Revert to base64 storage (not recommended)."""
    # Add back old columns
    op.add_column(
        'conversations',
        sa.Column('response_audio_base64', sa.Text(), nullable=True)
    )
    op.add_column(
        'conversations',
        sa.Column('audio_is_compressed', sa.Boolean(), nullable=True, server_default='true')
    )
    
    # Drop new column
    op.drop_column('conversations', 'response_audio_path')
    
    print("⚠️ Reverted to base64 storage - audio files on disk are orphaned")
