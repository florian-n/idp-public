# Setup

Create a conda environment and activate it:

```bash
conda create --name jaxidp python=3.11
conda activate jaxidp
conda install pip
```

Now, make sure all required libraries are available:

```bash
pip install -r requirements.txt
```

To be able to use the environment in Jupyter run

```bash
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

Install GPU JAX using

```bash
pip install -U "jax[cuda12]"
```

# Sources

- TIP3P model parameters: [TIP3P water model](https://docs.lammps.org/Howto_tip3p.html)
- Scattering lengths: [Neutron scattering lengths and cross sections](https://www.ncnr.nist.gov/resources/n-lengths/)
- Sassena: [Sassena](https://codebase.helmholtz.cloud/DAPHNE4NFDI/sassena)
