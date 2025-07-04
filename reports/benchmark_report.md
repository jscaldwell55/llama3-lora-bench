# LoRA Benchmark Report
Generated: 2025-07-04 22:12:52

## Summary
- Total configurations tested: 7
- Best memory efficiency: r=32 (2.7M params/GB)
- Fastest configuration: r=16 (31.1 samples/sec)

## Detailed Results

|   R |   Alpha |   Trainable Params |   Peak GPU (GB) |   Time (s) |   Final Loss |   Samples/sec |
|----:|--------:|-------------------:|----------------:|-----------:|-------------:|--------------:|
|   4 |       8 |    563200          |         3.10682 |   16.3113  |     0.849736 |       30.6536 |
|   8 |      16 |         1.1264e+06 |         3.1164  |   16.1686  |     0.739329 |       30.9242 |
|   8 |      16 |         1.1264e+06 |         2.60061 |    7.94906 |     1.19174  |       12.5801 |
|   8 |      16 |       nan          |       nan       |  nan       |   nan        |      nan      |
|   8 |      16 |       nan          |       nan       |  nan       |   nan        |      nan      |
|  16 |      32 |         2.2528e+06 |         3.13555 |   16.0598  |     0.639033 |       31.1337 |
|  32 |      32 |         9.0112e+06 |         3.37855 |   20.878   |     0.550973 |       23.9486 |