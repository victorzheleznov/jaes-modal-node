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