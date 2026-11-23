-- Model release history: every deploy, confirmation and rollback, in order.
-- The newest row per model is the release it serves; a restarted gateway
-- loads it. Numbered 006 so it does not clash with the open PRs' 003 to 005.

CREATE TABLE model_releases (
    seq bigserial PRIMARY KEY,
    at timestamptz NOT NULL,
    backend text NOT NULL,
    action text NOT NULL,                -- deployed, confirmed or rolled_back
    version text NOT NULL,               -- what the model serves after it
    from_version text,                   -- what it served before
    rollback_target text,                -- where a rollback would go after it
    actor text NOT NULL,                 -- operator or controller
    reason text NOT NULL
);

CREATE INDEX model_releases_backend_seq ON model_releases (backend, seq DESC);
