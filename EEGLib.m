
classdef EEGLib
    methods (Static)
        function sample = timeToSample(time_s, fs)
            sample = time_s * fs;
        end

        function time = sampleToTime(sample, fs)
            time = sample / fs;
        end

        function plotTriggers(signal, trigger_array)
            triggers = zeros(length(signal), 1);
            for x = 1:length(trigger_array)
                event = trigger_array(x);
                triggers(event) = 10;
            end
            hold on;
            plot(signal);
            plot(triggers);
            hold off;
        end

        function output = subdivide(signal, trigger_array, duration_s, pretrigger_s, fs)
            duration_samples = timeToSample(duration_s, fs);
            pretrigger_samples = timeToSample(pretrigger_s, fs);

            output = zeros(length(trigger_array), (abs(pretrigger_samples) + duration_samples));
            for x = 1:length(trigger_array)
                start = trigger_array(x);
                dice = signal((start + pretrigger_samples):(start + duration_samples - 1));  
                output(x, :) = dice;
            end
        end

        function filtered = filter_alpha(signal, sampling_rate)
            filtered = EEGLib.butterworth_filter(signal, 9/sampling_rate, 11/sampling_rate);
        end

        function filtered = filter_beta(signal, sampling_rate)
            filtered = EEGLib.butterworth_filter(signal, 18/sampling_rate, 22/sampling_rate);
        end

        function filtered = butterworth_filter(signal, f_low, f_cutoff)
            [b, a] = butter(5, [f_low, f_cutoff], 'bandpass');
            filtered = filter(b, a, signal);
        end

        function power = signal_power(signal)
            power = signal.^2;
        end

        function magnitude = time_frequency(signal, fs, window_size)
            magnitude = abs(spectrogram(signal, hann(window_size), window_size / 5, window_size));
        end
    end
end

