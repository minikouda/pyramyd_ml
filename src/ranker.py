def hybrid_score(sim, structured, w_sem=0.6):
    return w_sem * sim + (1 - w_sem) * structured

