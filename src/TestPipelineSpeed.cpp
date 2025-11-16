// TODO.
// See window configs.
// Timing budget per window for classification pipeline to ensure real time functionality:
// timing budget = hop (we need to be ready in time for next window which happens after hop)
// therefore timing budget = 0.85s (need to verify pipeline occurs within 850ms)