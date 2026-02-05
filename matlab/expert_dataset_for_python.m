%% export_dataset_for_python.m
% dataset_modes.mat (v7.3, timeseries 포함) -> dataset_flat.mat (v7, numeric arrays)
clear; clc;

inMat  = fullfile(pwd, "dataset_out", "dataset_modes.mat");
outMat = fullfile(pwd, "dataset_out", "dataset_flat.mat");

S = load(inMat, "dataset");
dataset = S.dataset;

% 채널 선택 (원하면 V 포함)
useV = false; % true로 하면 [I, HF, T, V]

runs = dataset.runs;
numRuns = numel(runs);

flat = struct();
flat.meta = dataset.meta;
flat.meta.exportedAt = char(datetime("now"));
flat.meta.useV = useV;

flat.runs = struct([]);

for k = 1:numRuns
    r = runs(k);

    t  = r.signals.I.Time(:);  % 시간축은 I의 Time 기준
    I  = double(r.signals.I.Data(:));
    HF = double(r.signals.HF.Data(:));
    T  = double(r.signals.T.Data(:));
    if useV
        V = double(r.signals.V.Data(:));
    end

    % 길이 맞추기(혹시 다르면 최소 길이로 자름)
    L = min([numel(t), numel(I), numel(HF), numel(T)]);
    if useV
        L = min(L, numel(V));
    end

    t  = t(1:L);
    I  = I(1:L);
    HF = HF(1:L);
    T  = T(1:L);
    if useV
        V = V(1:L);
    end

    % (T×C)로 스택
    if useV
        X = [I, HF, T, V];
        chNames = ["I","HF","T","V"];
    else
        X = [I, HF, T];
        chNames = ["I","HF","T"];
    end

    fr = struct();
    fr.mode  = r.mode;
    fr.label = r.label;
    fr.seed  = r.seed;
    fr.stopTime = r.stopTime;

    fr.t = t;
    fr.X = X;                 % numeric array (T×C)
    fr.channel_names = chNames;

    flat.runs = [flat.runs; fr];
end

% v7로 저장(파이썬에서 scipy.io.loadmat으로 바로 읽힘)
save(outMat, "flat", "-v7");
fprintf("[SAVED] %s\n", outMat);
