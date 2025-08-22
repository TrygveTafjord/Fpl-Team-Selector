# Fantasy Premier League Point Predictor

This project aims to create a point predictor and team selector for Fantasy Premier League (FPL). The predictor uses a Random Forest model to predict points based on player data from the five most recent matches and upcoming fixture information. Separate models are created for each player position (GK, DEF, MID, FWD).

## Data Sources

### Historical Player Data

Historical data for player performances is fetched from a publicly available dataset. The data includes various metrics collected over multiple seasons, providing the foundation for our model's predictions.

**Citation:**
```bibtex
@misc{anand2016fantasypremierleague,
  title = {{FPL Historical Dataset}},
  author = {Anand, Vaastav},
  year = {2022},
  howpublished = {Retrieved August 2022 from \url{https://github.com/vaastav/Fantasy-Premier-League/}}
}
```

### Real-Time Player and Fixture Data
Real-time data on players and upcoming fixtures is retrieved from the official Fantasy Premier League API. This data is used to inform the predictions for upcoming gameweeks and injuries.

Information on how to interact with the FPL API was obtained from this detailed guide: [Fantasy Premier League API Endpoints Guide](https://medium.com/@frenzelts/fantasy-premier-league-api-endpoints-a-detailed-guide-acbd5598eb19).

### Dependencies
Python (>= 3.7)
pandas
matplotlib
scikit-learn
requests


Equations behing the Bayesian Model
### Equations behind the Bayesian Model

**1. Likelihood**

The total points $y_{i}$ are modeled as a draw from a mixture distribution:

$y_{i} \sim (1-w_{i}) \cdot \text{NB}(\mu_{\text{blank},i}, \alpha_{\text{blank}}) + w_{i} \cdot \text{NB}(\mu_{\text{haul},i}, \alpha_{\text{haul}})$

Where: 
- $w_{i}$ is the probability of being in the "Haul" state for game i.
- $\mu_{\text{blank},i}$ and $\mu_{\text{haul},i}$ are the means of the Blank and Haul distributions, respectively.
- $\alpha_{\text{blank}}$ and $\alpha_{\text{haul}}$ are the dispersion parameters for the two distributions.

The Negative Binomial distribution is parameterized by its mean $\mu$ and dispersion $\alpha$.

**2. Link Functions (Regressions)**

The parameters $w_{i}$, $\mu_{\text{blank},i}$, and $\mu_{\text{haul},i}$ are dependent on the feature vector $X_{i}$ for each game via link functions.

**A. Haul Probability ($w_{i}$)**

A logistic regression (logit link function) is used to model the probability of a haul:

$\text{logit}(w_{i}) = \beta_{w,\text{intercept}} + X_{i}\beta_{w,\text{coeffs}}$

$w_{i} = \text{sigmoid}(\text{logit}(w_{i})) = \frac{1}{1+e^{-(\beta_{w,\text{intercept}}+X_{i}\beta_{w,\text{coeffs}})}}$

**B. Blank Mean ($\mu_{blank,i}$)**

The mean of the "Blank" distribution is modeled using a log link function (log to ensure positive mean for the negative bimodal distribution:

$\log(\mu_{\text{blank},i}) = \beta_{\text{blank,intercept}} + X_{i}\beta_{\text{blank,coeffs}}$

$\mu_{\text{blank},i} = \exp(\beta_{\text{blank,intercept}} + X_{i}\beta_{\text{blank,coeffs}})$

**C. Haul Mean ($\mu_{haul,i}$)**

The mean of the "Haul" distribution is modeled as an offset from the blank mean, also using a log link for the offset:

$\log(\text{offset}_{i}) = \beta_{\text{haul,intercept}} + X_{i}\beta_{\text{haul,coeffs}}$

$\mu_{\text{haul,i}} = \mu_{\text{blank,i}} + \exp(\log(\text{offset}_{i}))$

**3. Priors**

Priors are placed on all unknown parameters of the model.

**A. Regression Coefficients (β)**

Haul Probability Priors:

$\beta_{w, \text{intercept}} \sim \mathcal{N}(a_{1}, b_{1})$
$\beta_{w, \text{coeffs}} \sim \mathcal{N}(a_{2}, b_{2})$

Blank Mean Priors:

$\beta_{\text{blank}, \text{intercept}} \sim \mathcal{N}(\log(a_{3}), b_{3})$
$\beta_{\text{blank}, \text{coeffs}} \sim \mathcal{N}(a_{4}, b_{4})$

Haul Offset Priors:

$\beta_{\text{haul}, \text{intercept}} \sim \mathcal{N}(\log(a_{5}), b_{5})$
$\beta_{\text{haul}, \text{coeffs}} \sim \mathcal{N}(a_{6}, b_{6})$

**B. Dispersion Parameters (α)**

Blank Dispersion:

$\alpha_{\text{blank}} \sim \text{HalfNormal}(a_{7}, b_{7})$

Haul Dispersion:

$\alpha_{\text{haul}} \sim \text{HalfNormal}(a_{8}, b_{8})$
