import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# For pole-zero plotting, we'll just do a simple scatter in the z-plane.
# If you want a more sophisticated approach, there are specialized libs,
# but we'll keep it basic here.

# --------------------------
# 1) Define DTFT Functions
# --------------------------

def dtft_exponential(a, omega):
    """
    DTFT of x[n] = a^n u[n], for |a| < 1.
    X(e^{j omega}) = 1 / (1 - a e^{-j omega}).
    """
    return 1.0 / (1.0 - a * np.exp(-1j*omega))

def dtft_rect(L, omega):
    """
    DTFT of x[n] = 1 for n=0..L-1, 0 otherwise.
    X(e^{j omega}) = sum_{n=0}^{L-1} e^{-j omega n}
                    = e^{-j omega (L-1)/2} * sin(omega L/2) / sin(omega/2), (if using the closed-form).
    We'll just do it directly as the sum for demonstration.
    """
    # Be careful around omega = 0 (possible numerical issues).
    # For simplicity, let's just do the direct sum; for large L, consider vectorized or closed-form.
    # We'll handle the shape carefully if omega is an array.
    X = np.zeros_like(omega, dtype=np.complex128)
    for i, w in enumerate(omega):
        # sum_{n=0}^{L-1} e^{-j w n}
        # Could do partial fraction or direct formula, but let's do direct sum for clarity.
        n = np.arange(L)
        X[i] = np.sum(np.exp(-1j * w * n))
    return X

def dtft_damped_sinusoid(a, w0, omega):
    """
    DTFT of x[n] = (a^n) * cos(w0 * n) * u[n], for |a| < 1, damped sinusoid.
    x[n] = (a^n)(cos(w0 n)) => can be expressed as (a e^{j w0})^n + (a e^{-j w0})^n / 2.
    We'll compute it directly for demonstration.
    """
    # x[n] = Re{a^n e^{j w0 n}}. We'll just do a sum from n=0..infinity (theoretically),
    # but for plotting, let's do a closed form if possible:
    #   X(e^{j omega}) = DTFT{x[n]} = 0.5 [1/(1 - a e^{-j(w - w0)}) + 1/(1 - a e^{-j(w + w0)})]
    # as long as |a| < 1. We'll do that approach:
    # Make sure to handle array of omega carefully.
    numerator1 = 1.0
    numerator2 = 1.0
    # Denominators:
    # 1 - a e^{-j(omega - w0)}
    # 1 - a e^{-j(omega + w0)}
    # We compute them vectorized:
    exp_minus_jomega = np.exp(-1j * omega)
    # We'll combine exponents for the terms inside the denominators
    denom1 = 1.0 - a * np.exp(-1j*(omega - w0))  # (omega - w0)
    denom2 = 1.0 - a * np.exp(-1j*(omega + w0))  # (omega + w0)

    X = 0.5 * (numerator1/denom1 + numerator2/denom2)
    return X


# --------------------------
# 2) Helper: DFT approximation
# --------------------------

def dft_of_signal(x, N_fft):
    """
    Compute the N_fft-point DFT (FFT) of a finite signal x[n].
    Return:
      - freq bins (omega_k) in [0..2π)
      - X[k] complex array
    """
    X = np.fft.fft(x, n=N_fft)
    # digital freq axis, 0..2π
    omega_k = 2 * np.pi * np.arange(N_fft) / N_fft
    return omega_k, X


# --------------------------
# 3) Pole-Zero Plot function
# --------------------------
def plot_pole_zero_exponential(a, ax=None):
    """
    For x[n] = a^n u[n], the z-transform X(z) = 1 / (1 - a z^-1).
    Poles: z = a
    Zero: z = 0? Actually the system's factor is (1 - a z^-1) in the denominator => pole at z=a.
    We'll plot the unit circle and the single pole.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit circle')

    # Plot the pole at z=a (real axis if a is real)
    ax.plot([a], [0], 'rx', markersize=10, label='Pole at z=%0.2f' % a)

    # Adjust the axis
    ax.set_aspect('equal', 'box')
    max_radius = max(1.1, abs(a) + 0.1)
    ax.set_xlim([-max_radius, max_radius])
    ax.set_ylim([-max_radius, max_radius])
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title("Pole-Zero Plot (Exponential case)")
    ax.grid(True)
    ax.legend()


# --------------------------
# 4) Streamlit UI
# --------------------------

st.title("Extended DTFT Demonstration")

st.markdown("""
**Choose a signal** to see its DTFT and optionally a DFT-based approximation (finite sum + FFT).
""")

signal_type = st.selectbox(
    "Select signal type:",
    ["Exponential  (a^n u[n])",
     "Rectangular Pulse  (length L)",
     "Damped Cosine  (a^n cos(ω0 n) u[n])"]
)

# Sliders for relevant parameters
num_points = st.slider("Number of frequency samples for DTFT plot:", 100, 4000, 1024, 100)
show_dft = st.checkbox("Show DFT approximation?", value=False)

if signal_type.startswith("Exponential"):
    a = st.slider("Decay factor a (|a|<1)", 0.0, 0.99, 0.8, 0.01)
    # DTFT
    omega = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    X_dtft = dtft_exponential(a, omega)

    # Plot DTFT
    fig, axs = plt.subplots(2, 1, figsize=(7, 5))
    axs[0].plot(omega, np.abs(X_dtft), 'b')
    axs[0].set_title("|X(e^{jω})| (Exponential)")
    axs[0].set_xlabel("ω (rad/sample)")
    axs[0].set_ylabel("Magnitude")
    axs[0].grid(True)

    axs[1].plot(omega, np.angle(X_dtft), 'r')
    axs[1].set_title("∠X(e^{jω}) (Exponential)")
    axs[1].set_xlabel("ω (rad/sample)")
    axs[1].set_ylabel("Phase (radians)")
    axs[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    # Optional: DFT approximation
    if show_dft:
        st.markdown("### DFT Approximation")
        N_trunc = st.slider("Truncate signal length (for DFT):", 8, 256, 64, 8)
        # x[n] = a^n for n=0..N_trunc-1
        n = np.arange(N_trunc)
        x_trunc = a**n
        N_fft = st.slider("FFT size", N_trunc, 1024, max(256, N_trunc), 16)
        omega_k, X_dft = dft_of_signal(x_trunc, N_fft)

        fig2, ax2 = plt.subplots(2,1,figsize=(7,5))
        ax2[0].stem(omega_k, np.abs(X_dft), linefmt='b-', markerfmt='bo', use_line_collection=True)
        ax2[0].set_title("DFT Magnitude (Truncated signal)")
        ax2[0].set_xlabel("ω_k (rad/sample)")
        ax2[0].grid(True)

        ax2[1].stem(omega_k, np.angle(X_dft), linefmt='r-', markerfmt='ro', use_line_collection=True)
        ax2[1].set_title("DFT Phase (Truncated signal)")
        ax2[1].set_xlabel("ω_k (rad/sample)")
        ax2[1].grid(True)

        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown("""
        Notice how, with bigger truncation length and bigger FFT size, the DFT samples approach
        the **continuous** DTFT (blue/red line above).
        """)

    # Optional: Show pole-zero plot
    show_pz = st.checkbox("Show pole-zero plot (Exponential)?", value=False)
    if show_pz:
        fig3, ax3 = plt.subplots(figsize=(4,4))
        plot_pole_zero_exponential(a, ax3)
        st.pyplot(fig3)


elif signal_type.startswith("Rectangular"):
    L = st.slider("Rect pulse length L", 1, 100, 5, 1)
    # Compute DTFT
    omega = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    X_dtft = dtft_rect(L, omega)

    fig, axs = plt.subplots(2, 1, figsize=(7, 5))
    axs[0].plot(omega, np.abs(X_dtft), 'b')
    axs[0].set_title("|X(e^{jω})| (Rectangular Pulse)")
    axs[0].set_xlabel("ω (rad/sample)")
    axs[0].set_ylabel("Magnitude")
    axs[0].grid(True)

    axs[1].plot(omega, np.angle(X_dtft), 'r')
    axs[1].set_title("∠X(e^{jω}) (Rectangular Pulse)")
    axs[1].set_xlabel("ω (rad/sample)")
    axs[1].set_ylabel("Phase (radians)")
    axs[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    if show_dft:
        st.markdown("### DFT Approximation (actually same as the real signal for L samples!)")
        N_fft = st.slider("FFT size", L, 1024, max(64, L), 8)
        # The rectangular pulse is already finite length:
        x_trunc = np.ones(L)
        omega_k, X_dft = dft_of_signal(x_trunc, N_fft)

        fig2, ax2 = plt.subplots(2,1,figsize=(7,5))
        ax2[0].stem(omega_k, np.abs(X_dft), linefmt='b-', markerfmt='bo', use_line_collection=True)
        ax2[0].set_title("DFT Magnitude (Rect Pulse)")
        ax2[0].set_xlabel("ω_k (rad/sample)")
        ax2[0].grid(True)

        ax2[1].stem(omega_k, np.angle(X_dft), linefmt='r-', markerfmt='ro', use_line_collection=True)
        ax2[1].set_title("DFT Phase (Rect Pulse)")
        ax2[1].set_xlabel("ω_k (rad/sample)")
        ax2[1].grid(True)

        plt.tight_layout()
        st.pyplot(fig2)

elif signal_type.startswith("Damped"):
    st.markdown(r"Damped Cosine: $x[n] = a^n \cos(\omega_0 n) u[n],\ |a|<1.$")

    a = st.slider("Decay factor a (|a|<1)", 0.0, 0.99, 0.9, 0.01)
    w0 = st.slider("Cosine freq ω0 (rad/sample)", 0.0, 3.14, 1.0, 0.01)

    omega = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    X_dtft = dtft_damped_sinusoid(a, w0, omega)

    fig, axs = plt.subplots(2,1, figsize=(7,5))
    axs[0].plot(omega, np.abs(X_dtft), 'b')
    axs[0].set_title(f"|X(e^{chr(106)}ω)| (Damped Cosine: a={a}, w0={w0})")
    axs[0].set_xlabel("ω (rad/sample)")
    axs[0].set_ylabel("Magnitude")
    axs[0].grid(True)

    axs[1].plot(omega, np.angle(X_dtft), 'r')
    axs[1].set_title(f"∠X(e^{chr(106)}ω) (Damped Cosine)")
    axs[1].set_xlabel("ω (rad/sample)")
    axs[1].set_ylabel("Phase (radians)")
    axs[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    if show_dft:
        st.markdown("### DFT Approximation of the Damped Cosine")
        N_trunc = st.slider("Truncate length N:", 8, 256, 64, 8)
        n = np.arange(N_trunc)
        x_trunc = (a**n)*np.cos(w0*n)
        N_fft = st.slider("FFT size", N_trunc, 1024, max(256, N_trunc), 16)

        omega_k, X_dft = dft_of_signal(x_trunc, N_fft)

        fig2, ax2 = plt.subplots(2,1,figsize=(7,5))
        ax2[0].stem(omega_k, np.abs(X_dft), linefmt='b-', markerfmt='bo', use_line_collection=True)
        ax2[0].set_title("DFT Magnitude (Truncated damped cosine)")
        ax2[0].set_xlabel("ω_k (rad/sample)")
        ax2[0].grid(True)

        ax2[1].stem(omega_k, np.angle(X_dft), linefmt='r-', markerfmt='ro', use_line_collection=True)
        ax2[1].set_title("DFT Phase (Truncated damped cosine)")
        ax2[1].set_xlabel("ω_k (rad/sample)")
        ax2[1].grid(True)

        plt.tight_layout()
        st.pyplot(fig2)


# --------------------------
# Footer
# --------------------------
st.markdown("""
**Summary of Extended Ideas:**
- **Multiple signals**: exponential, rectangular pulse, damped sinusoid.
- **Parameter sliders**: control decay factor \a\, length \L\, sinusoid frequency \\\omega_0\.
- **DFT approximation**: shows how truncating a (possibly infinite) signal and taking the FFT 
  yields discrete frequency samples that approximate the continuous DTFT.
- **Pole-zero**: we illustrated it for the exponential case, 
  showing the pole at \z=a\ and the unit circle boundary.
""")
