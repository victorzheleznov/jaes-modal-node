---
layout: splash
classes:
  - wide
---

<h2 style="font-size: 1.5em" align="center">
  Stable Differentiable Modal Synthesis for Learning Nonlinear Dynamics
</h2>

<p style="font-size: 1.0em" align="center">
  Victor Zheleznov<sup>1</sup>, Stefan Bilbao<sup>1</sup>, Alec Wright<sup>1</sup> and Simon King<sup>2</sup>
</p>

<p style="text-align: center; font-size: 0.75em">
  <i>
    <sup>1</sup><a href="https://www.acoustics.ed.ac.uk/" target="_blank" rel="noopener noreferrer">Acoustics and Audio Group</a>, University of Edinburgh, Edinburgh, UK<br>
    <sup>2</sup><a href="https://www.cstr.ed.ac.uk/" target="_blank" rel="noopener noreferrer">Centre for Speech Technology Research</a>, University of Edinburgh, Edinburgh, UK<br>
  </i>
</p>

<p style="font-size: 1.0em; text-align: center">
  Accompanying web-page for the JAES submission
</p>

<div style="text-align: center; align-items: center">
  <a href="https://github.com/victorzheleznov/jaes-modal-node" class="btn btn--primary btn--small" target="_blank" rel="noopener noreferrer">
    Code
  </a>
</div>





## Abstract

Modal methods are a long-standing approach to physical modelling synthesis. Extensions to nonlinear problems are possible, including the case of a high-amplitude vibration of a string. A modal decomposition leads to a densely coupled nonlinear system of ordinary differential equations. Recent work in scalar auxiliary variable techniques has enabled construction of explicit and stable numerical solvers for such classes of nonlinear systems. On the other hand, machine learning approaches (in particular neural ordinary differential equations) have been successful in modelling nonlinear systems automatically from data. In this work, we examine how scalar auxiliary variable techniques can be combined with neural ordinary differential equations to yield a stable differentiable model capable of learning nonlinear dynamics. The proposed approach leverages the analytical solution for linear vibration of system's modes so that physical parameters of a system remain easily accessible after the training without the need for a parameter encoder in the model architecture. As a proof of concept, we generate synthetic data for the nonlinear transverse vibration of a string and show that the model can be trained to reproduce the nonlinear dynamics of the system. Sound examples are presented.





## Sound Examples

Below are some sound examples along with string and excitation parameters for the datasets used in the submission. All sound examples can be downloaded from [the accompanying repository](https://github.com/victorzheleznov/jaes-modal-node/tree/master/audio).





### Test Dataset

<table>
<thead>
<tr><th style="text-align: center">Linear</th><th style="text-align: center">Target</th><th style="text-align: center">Predicted</th><th style="text-align: center">$\gamma$</th><th style="text-align: center">$\kappa$</th><th style="text-align: center">$\nu$</th><th style="text-align: center">$x_{\mathrm{e}}$</th><th style="text-align: center">$x_{\mathrm{o}}$</th><th style="text-align: center">$f_{\mathrm{amp}}$</th><th style="text-align: center">$T_{\mathrm{e}}$</th><th style="text-align: left">Note</th></tr>
</thead>
<tbody style="white-space: nowrap">
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/40_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/40_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/40_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>232.5</td><td>1.05</td><td>174.4</td><td>0.12</td><td>0.90</td><td>4.9e+04</td><td>7.2e-04</td><td>Largest relative MSE for audio output (illustrated in the manuscript)</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/43_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/43_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/43_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>209.0</td><td>1.08</td><td>129.1</td><td>0.37</td><td>0.23</td><td>4.9e+04</td><td>1.2e-03</td><td>Smallest relative MSE for audio output</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/6_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/6_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/6_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>196.4</td><td>1.05</td><td>171.9</td><td>0.85</td><td>0.89</td><td>4.8e+04</td><td>1.3e-03</td><td>Strongest nonlinear effects (illustrated in the manuscript)</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/30_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/30_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/30_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>243.1</td><td>1.05</td><td>146.1</td><td>0.43</td><td>0.82</td><td>4.4e+04</td><td>1.4e-03</td><td>Random example #1</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/56_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/56_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/56_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>180.9</td><td>1.08</td><td>157.1</td><td>0.13</td><td>0.32</td><td>4.9e+04</td><td>1.4e-03</td><td>Random example #2</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/46_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/46_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/46_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>184.0</td><td>1.08</td><td>148.7</td><td>0.63</td><td>0.74</td><td>4.3e+04</td><td>6.6e-04</td><td>Random example #3</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/38_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/38_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/38_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>246.9</td><td>1.08</td><td>167.4</td><td>0.87</td><td>0.64</td><td>4.2e+04</td><td>9.9e-04</td><td>Random example #4</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/1_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/1_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/1_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>202.0</td><td>1.07</td><td>171.1</td><td>0.81</td><td>0.79</td><td>3.8e+04</td><td>1.1e-03</td><td>Random example #5</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/36_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/36_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/36_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>196.8</td><td>1.08</td><td>140.5</td><td>0.29</td><td>0.50</td><td>4.2e+04</td><td>1.5e-03</td><td>Random example #6</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/45_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/45_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/45_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>202.8</td><td>1.06</td><td>126.1</td><td>0.57</td><td>0.61</td><td>4.1e+04</td><td>5.3e-04</td><td>Random example #7</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/23_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/23_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/23_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>191.7</td><td>1.06</td><td>139.1</td><td>0.65</td><td>0.87</td><td>3.8e+04</td><td>1.3e-03</td><td>Random example #8</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/27_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/27_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/27_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>229.8</td><td>1.08</td><td>138.6</td><td>0.81</td><td>0.55</td><td>4.8e+04</td><td>1.1e-03</td><td>Random example #9</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/59_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/59_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823/59_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>190.5</td><td>1.08</td><td>125.2</td><td>0.87</td><td>0.58</td><td>4.8e+04</td><td>1.4e-03</td><td>Random example #10</td></tr>
</tbody>
</table>





### Validation Dataset

<table>
<thead>
<tr><th style="text-align: center">Linear</th><th style="text-align: center">Target</th><th style="text-align: center">Predicted</th><th style="text-align: center">$\gamma$</th><th style="text-align: center">$\kappa$</th><th style="text-align: center">$\nu$</th><th style="text-align: center">$x_{\mathrm{e}}$</th><th style="text-align: center">$x_{\mathrm{o}}$</th><th style="text-align: center">$f_{\mathrm{amp}}$</th><th style="text-align: center">$T_{\mathrm{e}}$</th><th style="text-align: left">Note</th></tr>
</thead>
<tbody style="white-space: nowrap">
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/3_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/3_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/3_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>203.2</td><td>1.07</td><td>171.0</td><td>0.83</td><td>0.38</td><td>4.7e+04</td><td>5.0e-04</td><td>Largest relative MSE for audio output</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/9_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/9_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/9_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>210.2</td><td>1.06</td><td>139.1</td><td>0.78</td><td>0.49</td><td>4.3e+04</td><td>1.4e-03</td><td>Smallest relative MSE for audio output</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/15_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/15_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/15_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>202.0</td><td>1.07</td><td>154.0</td><td>0.23</td><td>0.60</td><td>4.9e+04</td><td>1.4e-03</td><td>Random example #1</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/18_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/18_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/18_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>231.3</td><td>1.06</td><td>160.8</td><td>0.39</td><td>0.21</td><td>3.5e+04</td><td>9.6e-04</td><td>Random example #2</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/7_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/7_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/7_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>177.1</td><td>1.08</td><td>170.5</td><td>0.30</td><td>0.19</td><td>4.7e+04</td><td>1.2e-03</td><td>Random example #3</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/14_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/14_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/14_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>229.6</td><td>1.06</td><td>140.5</td><td>0.79</td><td>0.27</td><td>4.3e+04</td><td>8.0e-04</td><td>Random example #4</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/4_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/4_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_96000Hz_3sec_75modes_45f296f20960bb05ae8150b2eab8cf81/4_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>190.7</td><td>1.06</td><td>130.9</td><td>0.22</td><td>0.28</td><td>3.5e+04</td><td>1.4e-03</td><td>Random example #5</td></tr>
</tbody>
</table>





### Training Dataset

<table>
<thead>
<tr><th style="text-align: center">Linear</th><th style="text-align: center">Target</th><th style="text-align: center">Predicted</th><th style="text-align: center">$\gamma$</th><th style="text-align: center">$\kappa$</th><th style="text-align: center">$\nu$</th><th style="text-align: center">$x_{\mathrm{e}}$</th><th style="text-align: center">$x_{\mathrm{o}}$</th><th style="text-align: center">$f_{\mathrm{amp}}$</th><th style="text-align: center">$T_{\mathrm{e}}$</th><th style="text-align: left">Note</th></tr>
</thead>
<tbody style="white-space: nowrap">
<tr><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/57_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/57_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/57_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>169.5</td><td>1.03</td><td>169.9</td><td>0.72</td><td>0.31</td><td>2.5e+04</td><td>5.3e-04</td><td>Largest relative MSE for audio output</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/54_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/54_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/54_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>148.6</td><td>1.02</td><td>129.4</td><td>0.47</td><td>0.56</td><td>3.0e+04</td><td>1.3e-03</td><td>Smallest relative MSE for audio output</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/51_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/51_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/51_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>150.3</td><td>1.01</td><td>172.5</td><td>0.37</td><td>0.16</td><td>3.4e+04</td><td>1.5e-03</td><td>Random example #1</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/59_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/59_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/59_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>144.9</td><td>1.04</td><td>164.9</td><td>0.28</td><td>0.71</td><td>2.6e+04</td><td>1.0e-03</td><td>Random example #2</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/32_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/32_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/32_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>167.3</td><td>1.05</td><td>138.4</td><td>0.14</td><td>0.21</td><td>2.8e+04</td><td>8.1e-04</td><td>Random example #3</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/30_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/30_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/30_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>141.4</td><td>1.02</td><td>163.2</td><td>0.67</td><td>0.11</td><td>2.9e+04</td><td>7.5e-04</td><td>Random example #4</td></tr>
<tr><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/12_lin_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/12_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/sav_88200Hz_2sec_75modes_02b814394ad8deb51762b7949e8fb3c8/12_pred_PCM_24_0.1dBFS.wav" type="audio/wav"></audio></td><td>125.4</td><td>1.01</td><td>173.1</td><td>0.62</td><td>0.72</td><td>2.9e+04</td><td>7.2e-04</td><td>Random example #5</td></tr>
</tbody>
</table>