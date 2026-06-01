-- FAILSAFE Database Schema
-- PostgreSQL

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─── Users ────────────────────────────────────────────
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    name          VARCHAR(255) NOT NULL,
    role          VARCHAR(50) NOT NULL CHECK (role IN ('faculty','hod','admin')),
    department    VARCHAR(255),
    hashed_password TEXT NOT NULL,
    is_active     BOOLEAN DEFAULT true,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Uploads ──────────────────────────────────────────
CREATE TABLE uploads (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    uploaded_by   UUID REFERENCES users(id),
    filename      VARCHAR(255),
    semester      VARCHAR(50),
    total_students INT,
    high_risk     INT DEFAULT 0,
    medium_risk   INT DEFAULT 0,
    low_risk      INT DEFAULT 0,
    avg_risk      FLOAT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Student Predictions ──────────────────────────────
CREATE TABLE student_predictions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    upload_id       UUID REFERENCES uploads(id) ON DELETE CASCADE,
    student_index   INT NOT NULL,
    student_name    VARCHAR(255),
    risk_probability FLOAT NOT NULL,
    risk_level      VARCHAR(20) CHECK (risk_level IN ('High','Medium','Low')),
    shap_factors    JSONB,   -- top SHAP factors
    raw_features    JSONB,   -- original feature values
    grade_g1        FLOAT,
    grade_g2        FLOAT,
    absences        INT,
    studytime       INT,
    failures        INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_predictions_upload ON student_predictions(upload_id);
CREATE INDEX idx_predictions_risk ON student_predictions(risk_level);

-- ─── Interventions ────────────────────────────────────
CREATE TABLE interventions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id   UUID REFERENCES student_predictions(id) ON DELETE CASCADE,
    category        VARCHAR(100),
    action          TEXT,
    priority        VARCHAR(20),
    applied_by      UUID REFERENCES users(id),
    note            TEXT,
    status          VARCHAR(50) DEFAULT 'Applied',
    applied_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_interventions_pred ON interventions(prediction_id);

-- ─── Seed Demo Users ──────────────────────────────────
-- Passwords are bcrypt hashes of 'faculty123' and 'hod123'
INSERT INTO users (email, name, role, department, hashed_password) VALUES
('faculty@college.edu', 'Dr. Priya Sharma',  'faculty', 'Computer Science', '$2b$12$placeholder_hash_faculty123'),
('hod@college.edu',     'Prof. Rajesh Kumar', 'hod',     'Computer Science', '$2b$12$placeholder_hash_hod123');