# RetailMind AI --- 35-Day Incremental Roadmap

## Product Vision

Build one evolving customer-intelligence platform in which each daily
increment contributes a documented, testable capability. The roadmap
below is a planning framework. Only increments explicitly completed and
evidenced in the repository should be marked complete.

## Increment Register

  ----------------------------------------------------------------------------
                    Day Planned           Evidence /         Status
                        capability        deliverable        
  --------------------- ----------------- ------------------ -----------------
                      1 End-to-end ML     Logistic           Implemented in
                        pipeline and      Regression         notebook; verify
                        ML-as-a-Service   pipeline,          reproducibility
                        foundation        evaluation, Joblib 
                                          artifact           

                      2 Decision Tree,    Shared-split       In progress
                        Random Forest and comparison,        
                        SVM comparison    champion decision  

                      3 To be defined by  Notebook, code,    Planned
                        the learning plan tests and          
                                          documentation      

                      4 Feature           Feature/ensemble   Planned / confirm
                        engineering and   increment          exact scope
                        ensembles                            

                  5--35 Subsequent        Add only after     Planned
                        platform          scope is defined   
                        increments                           
  ----------------------------------------------------------------------------

## Definition of Done for Each Increment

Every increment should include: 1. A clear business or technical
objective. 2. A notebook or source-code implementation. 3. Reproducible
inputs and configuration. 4. Evaluation or tests appropriate to the
feature. 5. Persisted artifacts where relevant. 6. README and technical
documentation updates. 7. Git commit with a descriptive message. 8.
Known limitations and next steps.

## Release Discipline

-   Preserve earlier model artifacts and experiment reports.
-   Avoid silently changing the baseline.
-   Record breaking changes and dependency updates.
-   Promote a champion only after documented comparison.
-   Keep planned features separate from implemented capabilities.

## Future Product Direction

Potential platform capabilities may include feature engineering,
ensemble learning, model validation, explainability, experiment
tracking, model serving, monitoring, retraining workflows and
business-facing insights. These should be added only when defined in the
daily curriculum and implemented with evidence.
