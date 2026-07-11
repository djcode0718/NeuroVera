"""
SQLAlchemy ORM models for NeuroTriage database.

Defines the schema for:
- AnalysisRun: stores individual MRI analysis results
- CaseBankEntry: historical case bank entries with embeddings
- ReferenceNote: clinical reference notes for each tumor type
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Database URL - SQLite in project root
DATABASE_URL = "sqlite:///neurotriage.db"

# SQLAlchemy ORM base class
Base = declarative_base()

# Global engine and session factory (initialized via init_db)
engine = None
SessionLocal = None


class AnalysisRun(Base):
    """
    Stores metadata and results for a single MRI analysis run.
    
    Fields correspond to the design spec AnalysisRun model:
    - id: UUID primary key
    - created_at: timestamp of analysis start
    - status: one of "pending", "completed", "failed"
    - original_filename: name of uploaded file
    - top_class: predicted tumor class
    - top_confidence: confidence of top prediction
    - predictions_json: JSON dict of all class probabilities
    - gradcam_image_b64: base64-encoded PNG heatmap
    - feature_embedding: JSON array of 512-dim embedding
    - draft_report: final draft report text
    - routing: one of "auto-clear", "needs-review", "urgent"
    - justification: Orchestrator's routing justification
    - reasoning_trace_json: JSON array of agent trace entries
    - critic_revision_count: number of critic revisions (0-2)
    - models_used_json: JSON dict of {agent: model_name}
    - error_message: error details if status="failed"
    """
    __tablename__ = "analysis_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String, nullable=False)  # "pending", "completed", "failed"
    original_filename = Column(String, nullable=True)
    top_class = Column(String, nullable=True)
    top_confidence = Column(Float, nullable=True)
    predictions_json = Column(Text, nullable=True)  # JSON blob
    gradcam_image_b64 = Column(Text, nullable=True)  # base64 PNG
    feature_embedding = Column(Text, nullable=True)  # JSON array
    draft_report = Column(Text, nullable=True)
    routing = Column(String, nullable=True)  # "auto-clear", "needs-review", "urgent"
    justification = Column(Text, nullable=True)
    reasoning_trace_json = Column(Text, nullable=True)  # JSON array
    critic_revision_count = Column(Integer, nullable=False, default=0)
    models_used_json = Column(Text, nullable=True)  # JSON dict
    error_message = Column(Text, nullable=True)

    def __repr__(self):
        return f"<AnalysisRun(id={self.id}, status={self.status}, top_class={self.top_class})>"


class CaseBankEntry(Base):
    """
    Historical case entry in the case bank for retrieval-augmented analysis.
    
    Fields correspond to the design spec CaseBankEntry model:
    - id: UUID primary key
    - tumor_type: one of "glioma", "meningioma", "notumor", "pituitary"
    - confidence_at_insertion: confidence of original classification
    - summary: plain-language case description
    - feature_vector: JSON array of 512-dim embedding
    - source_file: reference to original file or dataset
    - created_at: timestamp of insertion
    """
    __tablename__ = "case_bank"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    tumor_type = Column(String, nullable=False)  # "glioma", "meningioma", "notumor", "pituitary"
    confidence_at_insertion = Column(Float, nullable=False)
    summary = Column(Text, nullable=False)
    feature_vector = Column(Text, nullable=False)  # JSON array of 512 floats
    source_file = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<CaseBankEntry(id={self.id}, tumor_type={self.tumor_type})>"


class ReferenceNote(Base):
    """
    Clinical reference notes for each tumor type.
    
    Fields correspond to the design spec ReferenceNote model:
    - id: UUID primary key
    - tumor_type: one of "glioma", "meningioma", "notumor", "pituitary"
    - note_text: the reference text content
    - source: citation or source of the note
    """
    __tablename__ = "reference_notes"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    tumor_type = Column(String, nullable=False)
    note_text = Column(Text, nullable=False)
    source = Column(String, nullable=False)

    def __repr__(self):
        return f"<ReferenceNote(id={self.id}, tumor_type={self.tumor_type})>"


def init_db() -> None:
    """
    Initialize the SQLite database and create all tables.
    
    This function:
    1. Creates a SQLite database at neurotriage.db in the project root
    2. Creates all tables defined by the Base models if they don't exist
    3. Sets up the global engine and session factory for use by the app
    
    Preconditions:
    - PROJECT_ROOT environment variable or current working directory is the project root
    
    Postconditions:
    - neurotriage.db file exists in the project root
    - All tables (analysis_runs, case_bank, reference_notes) are created
    - engine and SessionLocal globals are ready for use
    
    Example:
        from app.db.models import init_db, SessionLocal
        init_db()  # Creates neurotriage.db and tables
        session = SessionLocal()  # Get a session
    """
    global engine, SessionLocal

    # Create engine connecting to SQLite database in project root
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}  # Allow threading with SQLite
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Set up session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    """
    Get a new database session.
    
    Must be called after init_db() has been invoked.
    
    Returns:
        A SQLAlchemy session for database operations
        
    Example:
        session = get_db_session()
        runs = session.query(AnalysisRun).all()
        session.close()
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return SessionLocal()
