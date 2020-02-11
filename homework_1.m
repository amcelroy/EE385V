load('EEG.mat')

alpha_channel = EEGLib.filter_alpha(signal, fs);
beta_channel = EEGLib.filter_beta(signal, fs);

chunks = EEGLib.subdivide(signal, trigger, 6, -2, fs);
alpha_chunks = EEGLib.subdivide(alpha_channel, trigger, 6, -2, fs);
beta_chunks = EEGLib.subdivide(beta_channel, trigger, 6, -2, fs);

signal_pow = EEGLib.signalPowerChunks(chunks);
alpha_pow = EEGLib.signalPowerChunks(alpha_chunks);
beta_pow = EEGLib.signalPowerChunks(beta_chunks);

signal_fft = EEGLib.fftChunks(chunks, 250);
alpha_fft = EEGLib.fftChunks(alpha_chunks, 250);
beta_fft = EEGLib.fftChunks(beta_chunks, 250);

hz_60 = fs / 2;

signal_pow_avg = mean(signal_pow, 1);
alpha_pow_avg = mean(alpha_pow, 1);
beta_pow_avg = mean(beta_pow, 1);

signal_pow_var = std(signal_pow, 1, 1);
alpha_pow_var = std(alpha_pow, 1, 1);
beta_pow_var = std(beta_pow, 1, 1);

subplot(3, 1, 1);
plot(signal_pow_avg')
hold on;
plot(signal_pow_avg' + signal_pow_var')
plot(signal_pow_avg' - signal_pow_var')
hold off;
title('Raw signal Power, mean and mean +/- std')
ylabel('uVolts^2');
xlabel('Samples @ 250Hz')

subplot(3, 1, 2);
plot(alpha_pow_avg');
hold on;
plot(alpha_pow_avg' + alpha_pow_var');
plot(alpha_pow_avg' - alpha_pow_var');
hold off;
title('Alpha Channel Power, mean and mean +/- std')
ylabel('uVolts^2');
xlabel('Samples @ 250Hz')

subplot(3, 1, 3);
plot(beta_pow_avg');
hold on;
plot(beta_pow_avg' + beta_pow_var');
plot(beta_pow_avg' - beta_pow_var');
hold off;
title('Beta Channel Power, mean and mean +/- std')
ylabel('uVolts^2');
xlabel('Samples @ 250Hz')

% mvg_avg_signal_3 = EEGLib.movingAvg(chunks(20, :), 3);
% mvg_avg_signal_11 = EEGLib.movingAvg(chunks(20, :), 11);
% mvg_avg_signal_25 = EEGLib.movingAvg(chunks(20, :), 25);
% fft_signal = abs(EEGLib.fftSignal(chunks(20, :), hz_60));
% fft_mvg_avg_3 = abs(EEGLib.fftSignal(mvg_avg_signal_3, hz_60));
% fft_mvg_avg_11 = abs(EEGLib.fftSignal(mvg_avg_signal_11, hz_60));
% fft_mvg_avg_25 = abs(EEGLib.fftSignal(mvg_avg_signal_25, hz_60));

%%% Plot Several chunks on top of each other - Time Domain
% subplot(3, 1, 1);
% plot(abs(signal_fft(50:53, :)'));
% title('Signal 50 through 55 - Raw Fourier Domain');
% xlabel('Samples @ 250Hz');
% ylabel('Magnitude');
% ylim([0 800])
% legend('50', '51', '52', '53')
% 
% subplot(3, 1, 2);
% plot(abs(alpha_fft(50:53, :)'));
% title('Signal 50 through 55 - Alpha Fourier Domain');
% xlabel('Samples @ 250Hz');
% ylabel('Magnitude');
% ylim([0 800])
% legend('50', '51', '52', '53')
% 
% subplot(3, 1, 3);
% plot(abs(beta_fft(50:53, :)'));
% title('Signal 50 through 55 - Beta Fourier Domain');
% xlabel('Samples @ 250Hz');
% ylabel('Magnitude');
% ylim([0 800])
% legend('50', '51', '52', '53')

% %%% Plot Several chunks on top of each other - Time Domain
% subplot(3, 1, 1);/
% plot(chunks(50:53, :)');
% title('Signal 50 through 55 - Raw');
% xlabel('Samples @ 250Hz');
% ylabel('uV');
% legend('50', '51', '52', '53')
% 
% subplot(3, 1, 2);
% plot(alpha_chunks(50:53, :)');
% title('Signal 50 through 55 - Alpha');
% xlabel('Samples @ 250Hz');
% ylabel('uV');
% legend('50', '51', '52', '53')
% 
% subplot(3, 1, 3);
% plot(beta_chunks(50:53, :)');
% title('Signal 50 through 55 - Beta');
% xlabel('Samples @ 250Hz');
% ylabel('uV');
% legend('50', '51', '52', '53')


 x = 0;

%%% Plot Moving Average and FFT
% subplot(3, 1, 1);
% plot(chunks(20, :));
% hold on;
% plot(mvg_avg_signal_3);
% plot(mvg_avg_signal_11);
% plot(mvg_avg_signal_25);
% hold off;
% title('Trigger 20, 2 seconds before and 6 seconds after, Moving Averages'); 
% xlabel('Samples @ 250Hz');
% ylabel('uV');
% legend('Original', 'n = 3', 'n = 11', 'n = 25');
% 
% subplot(3, 1, 2);
% plot(fft_signal);
% hold on;
% plot(fft_mvg_avg_3);
% plot(fft_mvg_avg_11)
% plot(fft_mvg_avg_25);
% hold off;
% title('Trigger 20, FFT of Moving Averages to 120Hz'); 
% xlabel('Frequency in Hz');
% ylabel('Magnitude');
% legend('Original', 'n = 3', 'n = 11', 'n = 25');
% 
% subplot(3, 1, 3);
% plot(chunks(20, :) - mvg_avg_signal_3);
% hold on;
% plot(chunks(20, :) - mvg_avg_signal_11);
% plot(chunks(20, :) - mvg_avg_signal_25);
% hold off;
% title('Trigger 20, Difference between Raw and Moving Avg'); 
% xlabel('Samples @ 250Hz');
% ylabel('uV');
% legend('n = 3', 'n = 11', 'n = 25');

% signal_avg = mean(chunks, 1);
% alpha_avg = mean(alpha_chunks, 1);
% beta_avg = mean(beta_chunks, 1);
% 
% signal_var = std(chunks, 1, 1);
% alpha_var = std(alpha_chunks, 1, 1);
% beta_var = std(beta_chunks, 1, 1);
% 
% subplot(3, 1, 1);
% plot(signal_avg')
% hold on;
% plot(signal_avg' + signal_var')
% plot(signal_avg' - signal_var')
% hold off;
% title('Raw signal, mean and mean +/- std')
% ylabel('uVolts');
% xlabel('Samples @ 250Hz')
% 
% subplot(3, 1, 2);
% plot(alpha_avg');
% hold on;
% plot(alpha_avg' + alpha_var');
% plot(alpha_avg' - alpha_var');
% hold off;
% title('Alpha Channel, mean and mean +/- std')
% ylabel('uVolts');
% xlabel('Samples @ 250Hz')
% 
% subplot(3, 1, 3);
% plot(beta_avg');
% hold on;
% plot(beta_avg' + beta_var');
% plot(beta_avg' - beta_var');
% hold off;
% title('Beta Channel, mean and mean +/- std')
% ylabel('uVolts');
% xlabel('Samples @ 250Hz')


%Plot Raw with Triggers
% EEGLib.plotTriggers(signal, trigger)
% title('Overlay of Raw Signal and Triggers'); 
% xlabel('Samples @ 250Hz');
% ylabel('uV');
% legend('Raw', 'Triggers');

% Plot Raw, Alpha, Beta
% plot(chunks(20, :), 'LineWidth', 1);
% hold on;
% plot(alpha_chunks(20, :), 'LineWidth', 2);
% plot(beta_chunks(20, :), 'LineWidth', 2);
% hold off;
% 
% title('Trigger 20, 2 seconds before and 6 seconds after'); 
% xlabel('Samples @ 250Hz');
% ylabel('uV');
% legend('Raw', 'Alpha', 'Beta');

%Plot the power of the time domain signal
% plot(EEGLib.signal_power(chunks(20, :)), 'LineWidth', 1);
% hold on;
% plot(EEGLib.signal_power(alpha_chunks(20, :)), 'LineWidth', 2);
% plot(EEGLib.signal_power(beta_chunks(20, :)), 'LineWidth', 2);
% hold off;
% 
% title('Trigger 20, 2 seconds before and 6 seconds after Signal Power'); 
% xlabel('Samples @ 250Hz');
% ylabel('uV^2');
% legend('Raw', 'Alpha', 'Beta');


%Plot the power of the time domain signal
% plot(signal_avg, 'LineWidth', 1);
% hold on;
% plot(EEGLib.signal_power(alpha_chunks(20, :)), 'LineWidth', 2);
% plot(EEGLib.signal_power(beta_chunks(20, :)), 'LineWidth', 2);
% hold off;
% 
% title('Trigger 20, 2 seconds before and 6 seconds after Signal Power'); 
% xlabel('Samples @ 250Hz');
% ylabel('uV^2');
% legend('Raw', 'Alpha', 'Beta');