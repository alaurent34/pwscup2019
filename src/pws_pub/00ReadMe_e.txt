## About this directory
In this directory, you can find PWS Cup 2019 open information (ref Section 2.7 in the PWS Cup 2019 Contest Rule: https://www.iwsec.org/pws/2019/cup19-rule-e.pdf).

## Directory structure
- 00ReadMe_e.txt
  - This file

- Data/                               	Data
  - PWSCup2019_Osaka/                    PWSCup2019 artificial data (Osaka set: refer Section 2.7 in the contest rule paper)
    - info_region.csv                      Region assingment file
    - info_time.csv                        Time assignment file
    - orgtraces_teamXXX_dataYY_ZZZ.csv     Orignal trace for team XXX (data set YY)
    - reftraces_teamXXX_dataYY_ZZZ.csv     Reference trace for team XXX (data set YY)
  - PWSCup2019_Tokyo_Preliminary/        Region and time assignment to be used in the preliminary round
    - info_region.csv                      Region assignment file (refer Section 2.7.5 in the contest rule paper)
    - info_time.csv                        Time assignment file (refer Section 2.7.6 in the contest rule paper)

- Data_Anonymize/                      Anonymized trace set (before shuffling)

- Data_Anonymize_Shuffle/              Anonymized trace set (after shuffling)

- Data_IDDisclose/                     Result of ID disclure attacks

- Data_TraceInfer/                     Result of trace inference attacks

- Prog_Eval/                           Programs for evaluating utility/privacy score
  - EvalSecI.py                          Program for evaluating privacy score against ID disclosure (refer Section 2.7.3 in the contest rule paper)
  - EvalSecT.py                          Program for evaluating privacy score against trace inference (refer Section 2.7.4 in the contest rule paper)
  - EvalUtil.py                          Program for evaluating utility score (refer Section 2.7.2 in the contest rule paper)

- Prog_Shuffle/                        Program for shuffling an anonymized trace set
  - ShuffleIDs.py                        Program for shuffling user IDs in an anonymized trace set stored under the designated directory.

- SmplProg_Anonymize/                  Programs for anonymizing trace set (refer Section 2.7.1 in the contest rule paper)
  - A1-none.py                           Do nothing
  - A2-MRLH.py                           MRLH (Merging Regions and Location Hiding) [Shokri+,S&P11]
  - A3-kRR.py                            k-RR (k-ary randomized response) [Kairouz+,ICML16]
  - A4-PL.py                             PL (Planaer Laplace mechanism) [Andres+,CCS13]
  - A5-YA.py                             Yamaoka anonymization (cheating anonymization)

- SmplProg_IDDisclose/                 Programs for launching ID disclosure attacks (refer Section 2.7.1 in the contest rule paper)
  - I1-rand.py                           Randomly re-identify users
  - I2-VisitProb.py                      Re-identify users based on visit-probability vectors and by likelihood
  - I3-HomeProb.py                       Re-identify users based on visit-probability vectors and by likelihood (use only data at 8 o'clock)

- SmplProg_Track/                      Program for launching trace inference attacks (refer Section 2.7.1 in the contest rule paper)
  - T1-rand.py                           Randomly infer traces
  - T2-VisitProb.py                      Re-identify users based on the visit-probability vectors and likelihood, and then infer traces
  - T3-HomeProb.py                       Re-identify users based on the visit-probability vectors and likelihood, and then infer traces (use only data at 8 o'clock)

- Eval.bash                            Do the followings to the Osaka original trace set (team 001, data set 01): Anonymize (A2-MRLH), ID disclosure attack (I1-rand), and trace inference attack (T1-rand)

## Executing sample programs
Run on python 3.6 or later. Require numpy and scipy.
By running Eval.bash, you can test the anonymization, ID disclosure, and trace inference with the sample programs as follows:
- Eval.bash: Execute the following actions to the Osaka original trace set (team 001, data set 01)
  - Anonymization algorithm: A2-MRLH
  - ID disclosure algorithm: I1-rand
  - Trace inference algorithm: T1-rand

How to execute:
$ bash Eval.bash


## About the dataset
The location data below the Data directory is newly created based on the SNS-based People Flow Data ( https://nightley.jp/archives/1954/ ).
For more details, please see: PWS Cup 2019 Data Set ( https://www.iwsec.org/pws/2019/cup19-dataset-e.pdf ).

Attribute work to name in the SNS-based People Flow Data is as follows:

[Attribute work to name]
Nightley, Inc.
Shibasaki & Sekimoto Laboratory, the University of Tokyo
Micro Geo Data Forum
People Flow project
Center for Spatial Information Science at the University of Tokyo
