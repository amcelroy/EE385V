
classdef EEGLib
    methods (Static)
        function sample = timeToSample(time_s, fs)
            sample = time_s * fs;
        end
        
        function norm = minMaxNormalize(chunk)
            norm = (chunk - min(chunk)) / (max(chunk) - min(chunk));
        end
        
        function phases = phaseDiffChunks(chunks, fs)
           phases = zeros(size(chunks));
           for i = 1:size(chunks, 1)
              phases(i, :) = EEGLib.phase_diff(chunks(i, :), fs, 1); 
           end            
        end
        
        function phase = phase_diff(chunk, fs, plot_res)
            pretrig = chunk(:, 1:500);
            x = [0:499];
            x = 2*pi*x / fs;
            
            zci = @(v) find(v(:).*circshift(v(:), [-1 0]) <= 0);
            
            amp = max(pretrig) - min(pretrig);
            period = size(zci(pretrig), 1) / 4;
            phase = 0;
            

            fit = @(b, i)  b(1).*(sin(i.*b(2) + b(3)));    % Function to fit
            fcn = @(b) sum((fit(b, x) - pretrig).^2);                             % Least-Squares cost function
            s = fminsearch(fcn, [amp;  period;  phase]);
            sinewave = fit(s, x);
            
            xx = [0:1999];
            xx = 2*pi*xx / fs;
            sinewave = fit(s, xx);
            
            z1 = hilbert(chunk);
            instfrq = (2*pi)*unwrap(angle(z1));
            z2 = hilbert(sinewave);
            instfrq2 = (2*pi)*unwrap(angle(z2));
            phase = instfrq2 - instfrq;
            phase = (phase - min(phase))/ (max(phase) - min(phase));
            
            xxx = 1:2000;
            
            if(plot_res == 1)
                subplot(3, 1, 1)     
                plot(xxx, chunk, xxx, sinewave) 

                subplot(3, 1, 2)
                plot(xxx, angle(chunk), xxx, angle(sinewave))

                subplot(3, 1, 3)
                plot(phase)
            end
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