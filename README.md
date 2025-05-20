# rms-norm-transformers
Experimenting with precision of different rms norm transformers

## Float 32
Full precision RMS norm

## Float 16
- same loss and accuracy as float 32, but more memory used 

## Float 8
- Cannot implement with NVIDIA 3090
    - would require conversion from float 16 and extra computation
- is possible with other hardware like the nvidia h100

## Int 8
- Performance drops: more loss, less accuracy
- however, it uses the least memory

# Possible Improvements with Quantization
- Use QAT for RMSNorm-aware weight/activation adjustment.
    - QAT simulates low-precision arithmetic (e.g., int8) during training by inserting "fake quantization" operations (rounding + clamping) in the forward pass. Gradients are computed using high-precision values to maintain stability.