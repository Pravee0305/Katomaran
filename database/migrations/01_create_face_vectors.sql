-- Create face_vectors table to store FAISS-compatible face encodings
CREATE TABLE IF NOT EXISTS face_vectors (
    id SERIAL PRIMARY KEY,
    face_id INTEGER NOT NULL,
    encoding TEXT NOT NULL,  -- Base64 encoded face vector
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (face_id) REFERENCES face_encodings(id) ON DELETE CASCADE
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_face_vectors_face_id ON face_vectors(face_id);

-- Add comment for documentation
COMMENT ON TABLE face_vectors IS 'Stores face encoding vectors for FAISS similarity search';
COMMENT ON COLUMN face_vectors.encoding IS 'Base64 encoded numpy array of face encoding vector'; 