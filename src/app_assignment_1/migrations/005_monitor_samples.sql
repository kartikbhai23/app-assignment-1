-- Each sample in the failure and semantic monitor windows, so the gateway
-- can rebuild the windows after a restart. No message text.
-- Numbered 005 after #84's 003 and #99's 004; the runner applies whichever
-- files have not run.

CREATE TABLE monitor_samples (
    seq bigserial PRIMARY KEY,           -- insertion order, newest = highest
    at timestamptz NOT NULL,
    model text NOT NULL,
    monitor text NOT NULL,               -- failure or semantic
    outcome text NOT NULL,               -- failure: ok, format_failure, call_error
                                         -- semantic: agree, disagree, no_signal
    traffic text                         -- semantic only: own or rerouted
);

CREATE INDEX monitor_samples_window
    ON monitor_samples (model, monitor, traffic, seq DESC);
