# import the pyplot and wavfile modules
import os
import matplotlib
import matplotlib.pyplot as plot
import librosa
import librosa.display
import numpy as np
from scipy.io import wavfile

matplotlib.use('Qt5Agg')
# Read the wav file (mono)

# samplingFrequency, signalData = wavfile.read(r'D:\SPECTRON\QUALITITATIVE_RESULT\FF\EST_S0_Spectron.wav')
#
# # spec = librosa.stft(signalData, hop_length=512)
# # librosa.display.specshow(spec, sr=samplingFrequency, hop_length=512)
#
# # Plot the signal read from wav file
# # plot.figure()
# # librosa.display.specshow(spec, sr=samplingFrequency, hop_length=512, y_axis='log', cmap='viridis', x_axis='time')
# # plot.ylim([0,4000])
# # plot.show()
# plot.subplot(311)
# #
# plot.title('Spectrogram of a wav file with piano music')
# #
# plot.plot(signalData)
# #
# plot.xlabel('Sample')
# #
# plot.ylabel('Amplitude')
# #
# plot.subplot(312)
# #
# spec, freqs, t, im = plot.specgram(signalData, Fs=samplingFrequency)
# #
# plot.xlabel('Time')
# #
# plot.ylabel('Frequency')
# #
# plot.subplot(313)
# Z = 10. * np.log10(spec)
# Z = np.flipud(Z)
#
# NFFT = 256
# noverlap = 128
# Fs = samplingFrequency
# Fc = 0
# vmin = None
# vmax=None
#
# # padding is needed for first and last segment:
# pad_xextent = (NFFT-noverlap) / Fs / 2
# xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
# xmin, xmax = xextent
# freqs += Fc
# extent = xmin, xmax, freqs[0], freqs[-1]
# im2 = plot.imshow(Z, cmap='viridis', extent=extent, vmin=vmin, vmax=vmax, origin='upper')
# plot.axis('auto')
#
# # plot.imshow(spec)
# #
# plot.show()

# read gt
# gtsr, gtdata = wavfile.read(r'D:\SPECTRON\QUALITITATIVE_RESULT\FF\GT_S0.wav')
# esr, edata = wavfile.read(r'D:\SPECTRON\QUALITITATIVE_RESULT\FF\EST_S0_Spectron.wav')
# plot.subplot(311)
# specgt, freqs, t, im = plot.specgram(gtdata, Fs=gtsr)
# plot.subplot(312)
# spec_est, freqs_, t_, im_ = plot.specgram(edata, Fs=esr)
# plot.subplot(313)
# diff = np.abs(spec_est-specgt)
# # Z = 10. * np.log10(diff)
# Z = diff
# Z = np.flipud(Z)
#
# NFFT = 256
# noverlap = 128
# Fs = esr
# Fc = 0
# vmin = None
# vmax=None
#
# # padding is needed for first and last segment:
# pad_xextent = (NFFT-noverlap) / Fs / 2
# xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
# xmin, xmax = xextent
# freqs += Fc
# extent = xmin, xmax, freqs[0], freqs[-1]
# im2 = plot.imshow(Z, cmap='viridis', extent=extent, vmin=vmin, vmax=vmax, origin='upper')
# plot.axis('auto')
#
# plot.show()

## actual plot
FOLDERS = ["FF", "FM", "MF", "MM"]
ROOT_PATH = r'D:\SPECTRON\QUALITITATIVE_RESULT'

plot.figure()
plot.tight_layout()

for i, FOLDER in enumerate(FOLDERS):
    GT_PATH = os.path.join(ROOT_PATH, FOLDER, "GT_S0.wav")
    SPECTRON_EST_PATH = os.path.join(ROOT_PATH, FOLDER, "EST_S0_Spectron.wav")
    VCFILTER_EST_PATH = os.path.join(ROOT_PATH, FOLDER, "EST_S0_Vfilter.wav")

    gtsr, gtdata = wavfile.read(GT_PATH)
    esr_s, edata_s = wavfile.read(SPECTRON_EST_PATH)
    esr_v, edata_v = wavfile.read(VCFILTER_EST_PATH)

    plot.subplot(3, 4, 1+i)
    plot.axis("off")
    plot.tight_layout()
    plot.specgram(gtdata, Fs=gtsr)
    plot.subplot(3, 4, 5+i)
    plot.axis("off")
    plot.tight_layout()
    plot.specgram(edata_v, Fs=esr_v)
    plot.subplot(3, 4, 9+i)
    plot.axis("off")
    plot.tight_layout()
    plot.specgram(edata_s, Fs=esr_s)
    # plot.xlabel('Time')
    # plot.ylabel('Frequency')


plot.show()
