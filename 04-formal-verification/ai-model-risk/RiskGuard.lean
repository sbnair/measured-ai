/-
  RiskGuard.lean — a verified runtime guard for any position-sizing model.

  Positions are integers in basis points of the risk limit
  (10000 = 100% of the limit), so the arithmetic is exact.
-/

/-- Clamp a model's proposed position into [lo, hi]. -/
def riskGuard (lo hi y : Int) : Int := max lo (min hi y)

/-- The guarded position always respects the limits. -/
theorem guard_within (lo hi y : Int) (h : lo ≤ hi) :
    lo ≤ riskGuard lo hi y ∧ riskGuard lo hi y ≤ hi := by
  unfold riskGuard
  omega

/-- The guard never changes a position that is already inside the limits. -/
theorem guard_id (lo hi y : Int) (h1 : lo ≤ y) (h2 : y ≤ hi) :
    riskGuard lo hi y = y := by
  unfold riskGuard
  omega

/-- The guard keeps the model's ordering: a larger proposal never yields a smaller position. -/
theorem guard_mono (lo hi y y' : Int) (h : y ≤ y') :
    riskGuard lo hi y ≤ riskGuard lo hi y' := by
  unfold riskGuard
  omega

/-- Whatever the model is, wrapping it in the guard satisfies the limit for every input. -/
theorem guarded_model_safe {α : Type} (model : α → Int) (lo hi : Int) (h : lo ≤ hi) :
    ∀ x : α, lo ≤ riskGuard lo hi (model x) ∧ riskGuard lo hi (model x) ≤ hi :=
  fun x => guard_within lo hi (model x) h

-- A quick check on concrete numbers: a 13 000 bp proposal is cut to the 10 000 bp limit.
#eval riskGuard (-10000) 10000 13000
