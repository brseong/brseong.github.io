---
layout: post
title:  Questions about Machine Learning
categories: [Machine Learning]
---

# Questions
## Bishop, Pattern Recognition and Machine Learning

- 9.2.1
  - *Thus the maximization of the log likelihood function is not a well posed problem because such singularities will always be present and will occur whenever one of the Gaussian components ‘collapses’ onto a specific data point. Recall that this problem did not arise in the case of a single Gaussian distribution. To understand the difference, note that if a single Gaussian collapses onto a data point it will contribute multiplicative factors to the likelihood function arising from the other data points and these factors will go to zero exponentially fast, giving an overall likelihood that goes to zero rather than infinity. However, once we have (at least) two components in the mixture, one of the components can have a finite variance and therefore assign finite probability to all of the data points while the other component can shrink onto one specific data point and thereby contribute an ever increasing additive value to the log likelihood.*
  - For two data point on a 1-d Gaussian component, log-likelihood is
    $$\ln p(x_1,x_2)=\sum_{i=1}^{2}\ln\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{1}{2\sigma}(x_i-\mu)\right)\\
    =2\ln\frac{1}{\sigma\sqrt{2\pi}}-\frac{1}{2\sigma}(x_1-\mu)-\frac{1}{2\sigma}(x_2-\mu).$$
    Considering one data point $x_1$ is exactly the mean of the Gaussian $\mu$,
    $$=2\ln\frac{1}{\sigma\sqrt{2\pi}}-\frac{1}{2\sigma}(x_2-\mu).$$
    Then, as $\sigma\to0+$, $\ln p(x_1,x_2)\to -\infty.$ We can generalize the case to multivariate Gaussian.
    But, the book says that in Gaussian mixture, one Gaussian component can shrink into one data point, making the likelihood increase. Why does it happen?