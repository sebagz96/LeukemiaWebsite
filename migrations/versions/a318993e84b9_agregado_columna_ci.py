"""Agregado columna CI

Revision ID: a318993e84b9
Revises: 88ceb7d85c7e
Create Date: 2024-02-22 11:54:57.786135

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a318993e84b9'
down_revision = '88ceb7d85c7e'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('result',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('identification', sa.Integer(), nullable=True),
    sa.Column('patient_first_name', sa.String(length=100), nullable=True),
    sa.Column('patient_last_name', sa.String(length=100), nullable=True),
    sa.Column('image_path', sa.String(length=200), nullable=True),
    sa.Column('classification_result', sa.String(length=50), nullable=True),
    sa.Column('registration_date', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=200), nullable=True),
    sa.Column('password', sa.String(length=100), nullable=True),
    sa.Column('email', sa.String(length=100), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('user')
    op.drop_table('result')
    # ### end Alembic commands ###
