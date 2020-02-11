
classdef EEGLib
    methods (Static)
        function sample = timeToSample(time_s, fs)
            sample = time_s * fs;
        end

        function time = sampleToTime(sample, fs)
            time = sample / fs;
        end
        
        function moving_avg = movingAvgChunks(chunks, n)
            moving_avg = zeros(size(chunks));
            for x = 1:size(chunks, 1)
                moving_avg(x, :) = movmean(chunks(x, :), n);
            end
        end
        
        function moving_avg = movingAvg(signal, n)
            moving_avg = movmean(signal, n);
        end
        
        function complex_spectrum = fftChunks(chunks, range)
            if ~exist('range', 'var')
                range = (size(chunks, 2) / 2) + 1;
            end
            complex_spectrum = zeros(size(chunks, 1), range);
            for x = 1:size(chunks, 1)
                f = fft(chunks(x, :));
                complex_spectrum(x, :) = f(1:range);
            end
        end
        
        function complex_spectrum = fftSignal(signal, range)
            if ~exist('range', 'var')
                range = (size(signal, 2) / 2) + 1;
            end
            complex_spectrum = fft(signal);
            complex_spectrum = complex_spectrum(1:range);
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
            duration_samples = EEGLib.timeToSample(duration_s, fs);
            pretrigger_samples = EEGLib.timeToSample(pretrigger_s, fs);

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
        
        function power = signalPowerChunks(chunks)
            power = zeros(size(chunks));
            for x = 1:size(chunks, 1)
               power(x, :) = EEGLib.signal_power(chunks(x, :));
           end
        end

        function magnitude = time_frequency(signal, fs, window_size)
            magnitude = abs(spectrogram(signal, hann(window_size), window_size / 5, window_size));
        end
    end
end