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
  .text-small { font-size: 24px !important; line-height: 1.4; }
  .text-tiny  { font-size: 16px !important; line-height: 1.3; }
  .code-small { font-size: 14px !important; }

  .slide-vcenter {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    height: 100% !important;
  }
---

## Encoding models for fMRI

<div class="fit-width vcenter center">

  ![width:650px](resources/prf.png)

</div>


----
## Topographic organisation of neural representation

<div class="two-col">

<div class="col">

**Cortical organization** mirrors the **structure of stimulus spaces**, with adjacent neural populations encoding similar stimuli—be it spatial location, numerical proximity, or feature similarity.

</div>

<div class="col center vcenter">

  ![width:450px](resources/prf.png)

  de Hollander et al. (*in prep*)

</div>

</div>

----
## Topographic organisation of neural representation

<div class="two-col">

<div class="col">

**Cortical organization** mirrors the **structure of stimulus spaces**, with adjacent neural populations encoding similar stimuli—be it spatial location, numerical proximity, or feature similarity.

</div>

<div class="col center vcenter">

  ![width:480px](resources/odcs.png)

  de Hollander et al. (2021)


</div>

</div>


----
## Topographic organisation of neural representation

<div class="two-col">

<div class="col">

**Cortical organization** mirrors the **structure of stimulus spaces**, with adjacent neural populations encoding similar stimuli—be it spatial location, numerical proximity, or feature similarity.

</div>

<div class="col center vcenter">

  ![width:380px](resources/nprf.png)

  de Hollander et al. (in prep.)


</div>

</div>


---
## Encoding models

 - Encoding models allow us to map *features of the outside world* to *brain activation patterns*

   - Visuospatial location
   - Gabor orientations
   - Auditory frequency
   - Numerosity
   - Value
 - A "standard GLM" can be seen a very simple encoding model.

 ---
 #### Visuospatial mapping

<div class="two-col center vcenter">
<center>
<video controls width="700" autoplay loop>
  <source src="resources/prf_mapper.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
</center>
</div>

 ---
 #### Visuospatial mapping

<div class="center vcenter">

![width:700px](resources/prf_raw_timseries.png)

</div>

---
#### Standard PRF Model

<div class="two-col">
<div class="col">

Every voxel has a **Population Receptive Field (PRF)** characterized by:
- Preferred spatial location: $(\mu_x, \mu_y)$
- Width: $\sigma$
- Amplitude: $A$
- Baseline: $b$

</div>
<div class="col">


![height:500px](resources/prf_grid.png)


</div>
</div>

---
#### Standard PRF Model

<div class="two-col">

<div class="col text-medium">

The **encoding function** predicts the BOLD response $Y$ as the product of the PRF and the stimulus drive $S_i(X_i)$:

The **encoding function** predicts the BOLD response as the dot product of the PRF and stimulus drive $S_{x,y}$:

$$
f(S; \theta) = \sum_{x,y} \left[ A \cdot \exp\left(-\frac{(x - \mu_x)^2 + (y - \mu_y)^2}{2\sigma^2}\right) \cdot S_{x,y} \right] + b
$$

where $\theta = \{\mu_x, \mu_y, \sigma, A, b\}$.


</div>

<div class="col">

![height:500px](resources/prf_predictions1.png)

</div>
</div>

---
### Encoding models as your training ground

*Encoding models*, like the visuospatial PRF model lend themselve extremely well for fitting usin computational graphs/GPUs.

 * 7T data consists of ~ 300,000 voxels
 * Every time series ~250 time points
 * Simplest PRF model has 5 parameters
 * ~ 400 million variables

 * Without GPUs I couldn't do the research I do the way I do it.


----
### Assignment 2

<div class="two-col">

<div class="col">

*Implement a simple visuospatial PRF model using Tensorflow/Jax/Pytorch.*

`notebooks/3_implement_prf.ipynb`

</div>
<div class="col">

![height:500px](resources/prf_example.png)

</div>
</div>

---
#### Hint 1: Vectorize, vectorize, vectorize

<div class="text-medium">

Do *not* use for loops in plain Python

```python
predictions = []
for i in S.shape[0]: # loop over time
  pred = (S[i] * prf).sum(axis=[1, 2])
```

Use matrices (which can be heavily optimized, potentially on GPU):

```python
predictions = (S * prf).sum(axis=[1, 2])
```

**Use comments to remember which dimension is which**

```python
prf = get_prf(...) # (n_prfs, n_x_coordinates, n_y_coordinates)
```

</div>

---

##### Hint 2: Broadcasting

<div class="text-small">

```python
# Inputs:
#   S: (n_timepoints, n_x, n_y) = (238, 30, 30)
#   prf: (n_prfs, n_x, n_y) = (1000, 30, 30)

# Reshape for broadcasting:
S_expanded = S[:, tf.newaxis, :, :]      # Shape: (238, 1, 30, 30)
prf_expanded = prf[tf.newaxis, :, :, :]  # Shape: (1, 1000, 30, 30)

# Vectorized operation (e.g., element-wise multiply):
output = (S_expanded * prf_expanded).sum(axis=[2,3])  # Shape: (238, 1000)
```
**Memory savings**: 214M ($238 \times 1000 \times 30 \times 30$) → **1.1M** ($1000\times30\times30 + 238\times30\times30)$;99.5% reduction).

**Key Idea**
- **No `repmat`**: Use `tf.newaxis` to add singleton dimensions.
- **Broadcasting**: TensorFlow automatically expands dimensions for element-wise ops.
- **Efficiency**: Avoids copying data; computes on-the-fly.

---
### Why This Works
- `S_expanded` and `prf_expanded` align via broadcasting rules.
- **No memory blowup**: Only 2 small tensors are stored.
 - **Efficiency**: Avoids copying data; computes on-the-fly.
 - **XLA Optimization**: Fuses operations into a single computational graph, reducing overhead and accelerating execution.

</div>

---
### Hint 3: Start small


<div class='two-col text-medium'>


<div class="col">

**Select one voxel to *prototype***

```python
good_voxels = [82, 229, 538]

voxel_ts = v1_ts[82] # maybe do not do this
voxel_ts.shape
(258,)

voxel_ts = v1_ts[[82]] # do this
voxel_ts.shape
(258, 1)
```

</div>

<div class="col">

**Plot, plot, plot**

```python
good_voxels = [82, 229, 538]
v1_ts[good_voxels].plot()
sns.despine()
```

<div class='center'>

![height:275px](resources/prf_good_voxels.png)

</div>

</div>

</div>


---
### Hint 4: LLMs for the win

<div class="flex vcenter" style="display: flex !important;">

* Ask ChatGPT/Gemini/Le Chat/Claude to *help* you.
  * But maybe not do everything for you?
* In my experience: *You **need** to understand the code LLMs give you*

</div>

---
### Hint 5: Hemodynamic delay/temporal smoothing
<div class="two-col" style="align-items: center; height: 100%;">

<div class="col center vcenter" style="display: flex; flex-direction: column; justify-content: center;">

![width:500px](resources/hrf_plot.png)

$$
B(t) = (S * h)(t)
$$

</div>

<div class="col center vcenter" style="display: flex; justify-content: center; align-items: center;">
<video controls width="700" autoplay loop style="max-width: 100%; max-height: 50vh;">
  <source src="resources/stimuli_side_by_side_tight.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
</div>

</div>
