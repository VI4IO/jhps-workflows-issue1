# job-io-datasets
The German Climate Computing Center (DKRZ) maintains a monitoring system that gathers various statistics from the Mistral HPC system. Mistral has 3,340 compute nodes, 24 login nodes, and two Lustre file systems (lustre01 and lustre02) that provide a capacity of 52 Petabyte. 
The monitoring system is made up of open source components such as Grafana, OpenTSDB, and Elasticsearchtrebut also includes a lightweight self-developed data collector that captures continuously node statistics - we decided to implement an own collector when analyzing the overhead of existing approaches. 
Additionally, the monitoring system obtains various job meta-information from the Slurm workload manager and injects selected log files.

## Raw monitoring data

The monitoring system captures periodically I/O metrics on all client nodes, and sends them to a central database. 
The figure below illustrates the structure of the raw monitoring data using an example. 
In the example, data is captured on two nodes, on two file systems, for two metrics, and at nine time points ti , resulting in 4-dimensional data
(Node × File System × Metric × Time). 
The sizes of the node and the time dimensions are variable by nature. 
The other dimensions may be fixed for particular HPC systems, but we assumed they to be variable, to make the approach portable to other systems. 
Actually, on Mistral data is gathered every five seconds, for two Lustre file system, and for nine I/O metrics. 
Five of them (md read, md mod, md file create, md file delete, md other) capture metadata activities and the remaining four (read bytes, read calls, write bytes, write calls) capture data access.

![Data structure](assets/data_structure.png?raw=true "Data structure")

## Segmentation

We split the time series of each I/O metric into equal-sized time intervals (segments) and computes a mean performance for each segment. 
This stage preserves the performance units (e.g., Op/s, MiB/s) for each I/O metric. 
The example creates segments out of three successive time points just for illustration purposes. 
Depending on aggregation function, segments can be created of metrics (green boxes), of file systems (yellow boxes), of nodes (red boxes), or even over all dimensions (blue box). 
Actually, the real raw monitoring data is converted to ten minutes segments, which we found is a good trade-off to represent the temporal behavior of the application while it reduces the size of the time series.

## Categorization

Next, to get rid of the units, and to allow calculations between different I/O metrics, we introduced a categorization pre-processing step that takes into account the performance of the underlying HPC system and assigns a unitless ordered category to each metric segment. 
We use a three category system, which contains the LowIO=0, HighIO=1 and CriticalIO=4 categories. 
The category split points are based on the histogram of the obtained values, for any metrics, a segment with a value up to the 99%-Quantile it is considered to be LowIO, larger than the 99.9%-Quantile indicates CriticalIO, and between HighIO. 
This node-level data can then be used to compute job-statistics by aggregating across dimensions such as time, file systems, and nodes.

Datasets with more than 1.000.000 job I/O samples.

## Datasets

### job_metadata.csv
Anonymized job data

Hexadecimal and binary codings
### job_codings.csv
#### B-Coding 
Binary coding Binary coding represents monitoring data as a sequence of numbers, where each number represents the overall file system usage. 
The number is computed based on the nine metrics found in the segment, e.g., if a phase is read and write intensive it is encoded as one type of behavior. In this approach, each conceivable combination of activities has a unique number.
The approach maps the three categories to the following two states: The LowIO category is mapped to the non-active (0) state, and HighIO and CriticalIO categories are mapped to the active (1) state. On one side, by doing this, we lose information about performance intensity, but on other side, this simplification allows a more comprehensible comparison of job activities.

In our implementation, we use a 9-bit number to represent each segment, where each bit represents a metric. The bit is 1 if the corresponding metric is active, and 0 if not.
Translated to the decimal representation, metric segments can be coded as 1, 2, 4, 8, 16, and so on. Using this kind of coding we can compute a number for each segment, that describes unambiguously the file system usage, e.g., a situation where intensive usage of md read (Code=16) and read bytes (Code=32) occur at the same time and no other significant loads are registered is coded by the value 48. Coding is reversible, e.g., when having value 48, the computation of active metrics is straightforward.

To reduce the 4-dimensional data, we reduce that structure to two dimensions (segments metrics) by aggregating other dimensions by applying sum() function on score values. 
In the resulting table we leave zero scores, and change scores larger than zero to one. After coding
each segment, the jobs can be represented as a sequence of numbers, e.g., 
```
[1:5:0:0:0:0:0:0:96:96:96:96:96:96:96]
```
The monitoring dataset doesn’t provide information about what happens during the zero segments. 
It can be anything, like a job is waiting for resources, or computing something. 
It can also be that the job script cannot start immediately or run on a slow network. 
To catch such jobs, we aggregate multiple consecutive zero segments into one zero segment, thus the coding of the previous job would replace all 0:...:0 sequences just with 0, e.g., 
```
[1:5:0:96:96:96:96:96:96:96]
```

#### Q-Coding
This coding preserves monitoring data for each metric and each segment. 
As the name suggests, the value of a segment is converted into a hexadecimal number to allow creation of a string representing the I/O behavior. 
The numbers are obtained in two steps. 
Firstly, the dimension reduction aggregates the file system and the node dimensions and computes a mean value for each metric and segment, which lies in the interval [0,4].
Secondly, the mean values are quantized into NonIO + 16 I/O levels – 0 (NonI0) represents the interval [0,0.125), 1 [0.375,0.625), 2 [0.625,0.825) . . . , f [3.875,4]. 
The example below shows hexadecimal coding for a job containing 6 segments.

```
’q16_md_file_create’ : [0:0:2:2:2:9]
’q16_md_file_delete’ : [0:0:0:0:0:0]
’q16_md_mod’         : [0:0:0:0:0:0]
’q16_md_other’       : [0:0:0:0:0:0]
’q16_md_read ’       : [0:0:0:9:3:0]
’q16_read_bytes’     : [5:0:0:0:0:0]
’q16_read_calls’     : [0:0:0:0:0:0]
’q16_write_bytes’    : [0:0:0:0:f:f]
’q16_write_calls’    : [0:0:0:0:0:0]
```

### job_io_duration.csv
The IO-duration job profile contains the fraction of runtime, a job spent doing the individual I/O categories leading to 27 columns. 
The columns are named according to the following scheme: metric category, e.g, bytes read 0 or md file delete 4. 
The first part is the one of the nine metric names and the second part is the category number (LowIO=0, HighIO=1 and CriticalIO=4). 
These columns are used for machine learning as input features. 
There is a constraint for each metric (metric 0 + metric 1 + metric 4 = 1), that makes 9 features redundant, because they can be computed from the other features. So we have to deal with 18 features; max is 1.17.


### job_metrics.csv
  - Job-I/O-Balance: indicates how I/O load is distributed between nodes during job
runtime.
  - Job-I/O-Utilization: shows the average I/O load during I/O-phases.
  - Job-I/O-Problem-Time is the fraction of job runtime that is I/O-intensive; it is
approximated by the fractions of segments that are considered I/O intensive.


## Evaluation
The Run-Script `run.sh` in the root directory of the repository extracts datasets in the dataset and evaluation directories.
Then it clusters data and stores the results in the evaluation directory.
Finally it run analysis scripts and stores output in the evaluation directory.
