import jax

# Calibration needs float64 to resolve small parameter differences
jax.config.update("jax_enable_x64", True)
