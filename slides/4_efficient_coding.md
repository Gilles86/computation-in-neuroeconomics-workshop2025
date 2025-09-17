---
marp: true
theme: gaia
footer: Computation in Neuroeconomics seminar, Soglio, Switzerland, 2025
style: |
  h1 {
    font-size: 1.8em !important;
  }
  .two-col {
    display: flex !important;
    gap: 2rem !important;
    align-items: stretch !important;   /* same height columns */
    width: 100% !important;
  }
  .two-col > .col {
    min-width: 0 !important;           /* allow shrinking */
    flex: 1 1 0% !important;           /* default equal split */
  }

  /* Preset width splits */
  .two-col--50 > .col         { flex: 1 1 0% !important; } /* 50/50 */
  .two-col--30-70 > .col:first-child { flex: 3 1 0% !important; }
  .two-col--30-70 > .col:last-child  { flex: 7 1 0% !important; }
  .two-col--70-30 > .col:first-child { flex: 7 1 0% !important; }
  .two-col--70-30 > .col:last-child  { flex: 3 1 0% !important; }

  .col img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }

  /* Optional vertical centering for content inside a column */
  .vcenter {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
  }

  /* Handy helper */
  .center { text-align: center !important; }
  .fit {
    font-size: 0.8em; /* Base size, will scale down if needed */
  }
  .text-medium { font-size: 30px !important; line-height: 1.4; }
  .text-mediumsmall { font-size: 27px !important; line-height: 1.3; }
  .text-small { font-size: 24px !important; line-height: 1.3; }
  .text-twenty  { font-size: 20px !important; line-height: 1.3; }
  .text-tiny  { font-size: 16px !important; line-height: 1.3; }
  .code-small { font-size: 14px !important; }

  .slide-vcenter {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    height: 100% !important;
  }

---

### Efficient coding, efficiently coded

<div class="center">

![width:350px](resources/estimation_response.png)

</div>

**Gilles de Hollander**

---
### Efficient Coding: Key Ideas

**1. Sensory Space Representation**
- Stimulus features encoded in sensory space
- Noise is **homoscedastic** (constant variance)
- Encoding function:
  e,g., $f(x) = x^\alpha$ ($\alpha < 1$)
- Neurocognitive representation $r$:
  $$r|x_0 \sim \mathcal{N}(f(x_0), \nu^2)$$

---
### Efficient coding: Key ideas 
**2. Bayesian Inference**
- Estimate stimulus value $\hat{x}$ using Bayesian inference:
- Posterior distribution:
  $$p(x|r) = \frac{p(r|x)p(x)}{p(r)}$$
 - Posterior mean is least-square estimator
  $$\hat{x} = \mathbb{E}[x|r] = \int x \, p(x|r) \, dx$$

---
#### Why computational graphs?


 * For model fitting we want to estimate parameters.
 * Often we can not derive likelihood functions, but we can *evaluate* them for specific values.
   * Approximate integrals using grids (GPU!) 
   * MCMC sampling
 * Both often involve the same calculation on a very large number of variables.

---
#### Implementation Steps


<div class="text-small">


<div class="two-col">
<div class="col">

**1. Define the Generative Model**
- **Encoding function:** $f(x) = x^\alpha$
- **Noise model:** $r \sim \mathcal{N}(f(x_0), \nu^2)$

**2. Build the Likelihood Grid**
- Create a grid of possible $x_0$ and $r$ values
- Make a $p(r | x_0)$ for each pair

</div>

<div class="col">

**3. Bayesian Inference**
- For each observed $r$, compute the posterior:
  $$p(x|r) = \frac{p(r|x)p(x)}{p(r)}$$
- Estimate $\hat{x}$ as the expected value:
  $$\hat{x} = \mathbb{E}[x|r]$$

**4. Data Likelihood Function**
- Define a function that returns the response distribution over $\hat{x}$ for any $x_0$: $p(\hat{x} | x_0)$.
</div>
</div>

---
### Approach
Approximate (bounded) distributions using large arrays (vectorize, vectorize, vectorize).

---
### Assignment 5: Efficient coding
 - We wil now go over some code together in `notebooks/5_efficient_coding.ipynb`

