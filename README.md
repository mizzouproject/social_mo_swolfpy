# social_mo_swolfpy
This repository extends swolfpy by adding social criteria and multi-objective optimization.

This project prepares an extension of swolfpy as developed by Sardarmehni, et al. (2022). The extension allows swolfpy to consider social metrics, such that, besides optimizing total costs or an environmental impact, it can also select the social metrics as an optimization criterion. Social metrics are modelled in $/Mg and follow a similar approach to the swolfpy costs LCIA method. 
Two social metrics are proposed: Worker’s Physical Exposure and Turnover. 
Full details on this extension as well as explanations of the case study conducted to test such metrics will be available soon as a research paper. 

## Reference
Note: this project follows Sardarmenhi, et al. (2022) swolfpy development. For more information about swolfpy refer to *Sardarmehni, M., Anchieta, P. & Levis, J. (2022.) Solid Waste Optimization Life-cycle Framework in Python (SwolfPy). Journal of Industrial Ecology.*
Or visit the GitHub project space at https://github.com/SwolfPy-Project

## What is new?
Swolfpy module is based on Sardarmehni, et al. (2022). Add the social metrics values as new activities in the technosphere database.

## Branches
- version_1.0 : includes metrics for process models LF, MRF, Comp, WTE and AD.
- withcollection_v1 : social metrics for process models added in version_1.0 plus collection process


