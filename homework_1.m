alpha_channel = EEGLib.filter_alpha(signal, fs);
beta_channel = EEGLib.filter_beta(signal, fs);

hold on;
plot(signal);
plot(alpha_channel + 20);
plot(beta_channel + 40);
hold off;
