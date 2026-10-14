# Task 2 architecture

## Foundation status

Milestone 0 establishes an installable Python package and collaboration conventions. There is no application logic or data contract yet. No assumptions about the dataset schema, sampling rate, coordinates, or event clock have been validated.

## Responsibility boundaries

| Responsibility | Future responsibility | Must not own |
| --- | --- | --- |
| Ingestion | Discover original files by match ID; read validated source formats | Metric calculation or rewriting sources |
| Validation | Verify structure, identifiers, units, clocks, coordinates, and completeness | Silent repair of invalid input |
| Preprocessing | Produce documented derived representations | Modifying originals or concealing missingness |
| Synchronization | Establish explicit tracking/event timeline mappings | Assuming identical clocks or period conventions |
| Analytics | Calculate documented metrics from validated inputs | Inventing physiological labels |
| Playback | Consume prepared positions and synchronized events for a later board | Parsing original dataset files inside UI code |

These are design boundaries, not modules to create now. Implement ordinary Python functions/classes as needed before introducing services or plugin frameworks.

## Multiple matches and source discovery

Future entry points must accept a match ID and configurable data root. `3812` is a development example only. Preserve the dataset's original directory hierarchy and filenames when placing it under the local raw root. Inventory its actual naming conventions and discover the corresponding tracking, event, roster, and metadata files without renaming them. Resolve global reference files separately. Fail clearly on missing files or multiple unresolved candidates.

Do not prescribe source extensions, column names, or source subdirectories before inspection. Derived artifacts may later use a project-owned layout such as `data/processed/<match-id>/<processing-version>/`; this example is not a source dataset schema. Each run should identify its source files/hashes, configuration, software version, validation findings, and outputs.

## Synchronization and coordinate decisions

Before synchronization, verify time units, period resets, stoppages, timestamp origins, event precision, and tracking coverage. Before distances or pitch rendering, verify coordinate units, pitch dimensions, origin, axes, and team direction by period. Document any transformation and retain links to original time and position values.

## Future integrations

Video CV and physiological inputs will have their own validated ingestion and time mapping. Neither is available now. Keep domain calculations independent of future transport/UI choices, allowing a tactical board to consume prepared match data without embedding ingestion logic. Prediction, injury-risk modeling, and genetics remain separate future research scopes.

## Next milestone gate

After authorization and lawful data access: inventory actual files, verify provenance and schemas, identify match/global relationships, and document unknowns. Only then approve a concrete ingestion design and meaningful data tests.
