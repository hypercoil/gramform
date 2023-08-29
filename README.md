# ``gramform``
A rudimentary tree-based grammar for 'compiling' string-to-function formulae

``gramform`` is a small, extensible Python library for parsing and evaluating string-to-function formulae. It is designed to be used in conjunction with other libraries, such as ``pandas`` and ``nibabel``, to provide a simple, intuitive interface for evaluating mathematical expressions.

This library was designed to undergird the ``hypercoil`` library and its supporting pillars, although ``gramform`` can be used independently. ``hypercoil`` is a work-in-progress system that brings differentiable programming to the field of brain mapping via functional magnetic resonance imaging (fMRI). ``gramform`` is used to parse and evaluate mathematical expressions in several settings within the ``hypercoil`` system.

Data frame manipulation operations (``dfops``) are one such setting, where ``gramform`` combines a (currently incomplete and rather brittle) ``R``-inspired Wilkinson-like formula system with (fairly robust) domain-specific functions that facilitate neuroimaging applications, such as confound model specification.

Another setting is voxelwise operations on images (``imops``), where ``gramform`` is used to implement a subset of simple GPU-accelerated image mathematics operations that is eventually intended to be comparable with systems such as ``AFNI``'s ``3dcalc``, ``FSL``'s ``fslmaths``, or ``ANTs``' ``ImageMath``. At the moment, ``imops`` is mostly limited to mask manipulation and voxelwise arithmetic, but it is intended to be extended to include more complex operations, such as spatial filtering and voxelwise regression.

Finally, a third setting (currently available only in the ``hypercoil`` library itself) is the addressing of neural network parameters (``nnops``), where ``gramform`` supports targeting ``hypercoil``'s initialisation and parameterisation operations on specific tensor attributes of ``equinox`` modules.

## Installation
``gramform`` is available on PyPI and can be installed with ``pip``:
```bash
pip install gramform
```

## Basic usage
``gramform`` is designed to be used in conjunction with other libraries, such as ``pandas`` and ``nibabel``, to provide a simple, intuitive interface for evaluating mathematical expressions. The following examples demonstrate some of the basic functionality of ``gramform``.

### Confound model specification

``gramform`` can be used to specify confound models for neuroimaging applications. The following example demonstrates how to specify a confound model for a simple linear regression, using the ``R``/``patsy``-inspired Wilkinson-like formula syntax augmented with some domain-specific shorthand:

```python
from gramform.dfops import ConfoundFormulaGrammar
from pkg_resources import resource_filename
import json
import pandas as pd


# Begin helper functions to load data
# -----------------------------------
def path_from_examples(fname):
    return resource_filename('gramform', f'_resources/{fname}')

def read_json(path):
    with open(path) as file:
        metadata = json.load(file)
    return metadata

confdata = path_from_examples('desc-confounds_timeseries.tsv')
confmeta = path_from_examples('desc-confounds_timeseries.json')
df = pd.read_csv(confdata, sep='\t')
metadata = read_json(confmeta)
# -----------------------------------
# End helper functions to load data


# Define a grammar for confound model specification.
grammar = ConfoundFormulaGrammar()

# Compile and apply a 36-parameter model to the data.
# ----------------------------------------------------------------------------
# The 36-parameter model is a standard model for fMRI confound regression
# that includes 9 base time series: 6 motion parameters, and mean white
# matter, CSF, and global signal time series. Each base time series is
# augmented with its first derivative, and the resulting 18 time series are
# squared and concatenated to form the final model.
#
# Note the shorthand notation for rps, wm, csf, and gsr, which are
# automatically expanded to the BIDS-standard regressor names. The ddx and ^^x
# operators are also shorthand for the first derivative and squaring
# operations applied inclusively so that the original regressor is included
# as well. (To avoid inclusion of the original regressor, you would use the dx
# and ^x operators instead.)
model_36 = '(dd1(rps + wm + csf + gsr))^^2'
f_36 = grammar.compile(model_36)
out_36, meta_36 = f_36(df, metadata)

# Compile and apply a 36-parameter model augmented with spike regressors.
# ----------------------------------------------------------------------------
# The spike regressors are created first by identifying time points where the
# framewise displacement (fd) or DVARS (dv) exceeds a threshold (0.5 mm and
# 1.5, respectively). We use an indicator function `1_[param](regressor)` to
# create a binary time series that is 1 when the regressor exceeds the
# threshold and 0 otherwise. We then use the OR operator to combine the two
# binary time series into a single time series that is 1 when either the fd or
# dv regressor exceeds its threshold. Finally, we use the SCATTER operator to
# create a separate "spike" time series for each time point where the
# combined regressor is 1.
model_36spk = model_36 + ' + [SCATTER]([OR](1_[>0.5](fd) + 1_[>1.5](dv)))'
f_36spk = grammar.compile(model_36spk)
out_36spk, meta_36spk = f_36spk(df, metadata)

# Compile and apply an anatomical CompCor model.
# ----------------------------------------------------------------------------
model_acc = 'v_{{29.9; acc; Mask=CSF,WM}}'
f_acc = grammar.compile(model_acc)
out_acc, meta_acc = f_acc(df, metadata)

# Compile and apply an ICA-AROMA model.
# ----------------------------------------------------------------------------
model_aroma = 'wm + csf + {{aroma; MotionNoise=True}}'
f_aroma = grammar.compile(model_aroma)
out_aroma, meta_aroma = f_aroma(df, metadata)
```

Note that this system is not particularly "smart" or robust at this time, and its use outside of the specific context of confound modelling for fMRI is not currently recommended. Instead, a more mature framework such as ``patsy`` or ``formulaic`` should be used.

### Image mathematics

``gramform`` can be used to perform simple image mathematics operations. The following example demonstrates how to use ``gramform`` to perform voxelwise arithmetic on images (requires ``nibabel``, ``jax``, and for this example ``templateflow``):

```python
from gramform.imops import (ImageMathsGrammar, NiftiFileInterpreter)
import nibabel as nb
import templateflow.api as tflow

# Get GM, WM, and CSF probabilistic segmentations from the MNI152NLin2009cAsym
# template.
gm = tflow.get(
    'MNI152NLin2009cAsym', resolution=2, suffix='probseg', label='GM'
)
wm = tflow.get(
    'MNI152NLin2009cAsym', resolution=2, suffix='probseg', label='WM'
)
csf = tflow.get(
    'MNI152NLin2009cAsym', resolution=2, suffix='probseg', label='CSF'
)

# Define a grammar for image mathematics.
# ----------------------------------------------------------------------------
# The standard interpreter for the grammar is the NiftiObjectInterpreter,
# which interprets all operands as NiftiImage objects. Because we're using
# paths to Nifti files instead, we need to use the NiftiFileInterpreter.
grammar = ImageMathsGrammar(default_interpreter=NiftiFileInterpreter())

# Get the union of the p > 0.9 WM and CSF masks.
model_wmcsf = '(IMGa -bin[0.9]) -or (IMGb -bin[0.9])'
f_wmcsf = grammar.compile(model_wmcsf)
nifti_wmcsf = f_wmcsf(wm, csf)
nifti_wmcsf.to_filename('/tmp/wmcsf.nii.gz')


# Get the value of the GM probability map outside of the dilated union of the
# p > 0.9 WM and CSF masks.
model_gm = 'IMGa -mul (((IMGb -bin[0.9]) -or (IMGc -bin[0.9])) -dil[1] -neg)'
f_gm = grammar.compile(model_gm)
nifti_gm = f_gm(gm, wm, csf)
nifti_gm.to_filename('/tmp/gm.nii.gz')
```

Note that this system has some significant limitations, notable among them the inability to recognise an argument that occurs twice in the same expression. For example, the following expression will NOT behave as expected:

```python
model_fail = 'IMGa -mul ((IMGa -bin[0.9]) -or (IMGb -bin[0.9]))'
f_fail = grammar.compile(model_fail)
nifti_fail = f_fail(gm, wm)
nifti_fail.to_filename('/tmp/fail.nii.gz')
```

Similarly, under the hood, the 'compiled' function does not expect or handle keyword arguments. You can work around these limitation by providing the same argument twice when calling the function, or by wrapping the call in a parent function that duplicates the argument. For example:

```python
def f_fail_wrap(IMGa, IMGb):
    return f_fail(IMGa, IMGa, IMGb)
```
