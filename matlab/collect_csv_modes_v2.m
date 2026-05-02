%% collect_csv_modes.m
clear; clc;

%% ===== 사용자 설정 =====
modelName = "main_model";
stopTime = 10;
numRunsPerMode = 10;
saveFolder = fullfile(pwd, "dataset_out");
modeBlockNameHint = "mode";

% 수집할 mode
% 현재 검출 기준:
% Mode 0,1,2 = 비검출
% Mode 3,4,7 = 검출
modeList = [0 1 2 3 4 7];

%% ===== 저장 폴더 =====
if ~exist(saveFolder, "dir")
    mkdir(saveFolder);
end

%% ===== 모델 로드 =====
load_system(modelName);

% 혹시 실행 중이면 정지
try
    set_param(modelName, "SimulationCommand", "stop");
    pause(1);
catch
end

% Fast Restart는 디버깅 단계에서는 끔
try
    set_param(modelName, "FastRestart", "off");
catch
end

%% ===== MODE 블록 찾기 =====
modeBlk = findModeBlock(modelName, modeBlockNameHint);
fprintf("[INFO] MODE block found: %s\n", modeBlk);

%% ===== 실행 =====
for m = modeList
    for runIdx = 1:numRunsPerMode
        fprintf("\n=============================\n");
        fprintf("[RUN] mode=%d, run=%d/%d\n", m, runIdx, numRunsPerMode);
        fprintf("=============================\n");

        % mode 설정
        set_param(modeBlk, "Value", num2str(m));

        % 재현성용 시드
        rng(1000*m + runIdx, "twister");

        % 혹시 실행 중이면 정지
        try
            set_param(modelName, "SimulationCommand", "stop");
            pause(0.5);
        catch
        end

        % 시뮬레이션
        simOut = sim(modelName, ...
            "StopTime", num2str(stopTime), ...
            "ReturnWorkspaceOutputs", "on");

        % 디버그
        disp("[DEBUG] simOut contains:");
        disp(simOut.who)

        % ===== 신호 읽기 =====
        [I_data, HF_energy_data, HF_rms_data, hf_norm_data, ...
            arc_flag_data, arc_detect_data, T_data, t_data] = readSignals(simOut);

        % 길이 맞추기
        N = min([
            numel(I_data), numel(HF_energy_data), numel(HF_rms_data), ...
            numel(hf_norm_data), numel(arc_flag_data), numel(arc_detect_data), ...
            numel(T_data), numel(t_data)
        ]);

        I_data          = I_data(1:N);
        HF_energy_data  = HF_energy_data(1:N);
        HF_rms_data     = HF_rms_data(1:N);
        hf_norm_data    = hf_norm_data(1:N);
        arc_flag_data   = arc_flag_data(1:N);
        arc_detect_data = arc_detect_data(1:N);
        T_data          = T_data(1:N);
        t_data          = t_data(1:N);

        % boolean/logical이면 CSV 저장을 위해 double로 변환
        arc_flag_data   = double(arc_flag_data);
        arc_detect_data = double(arc_detect_data);

        % mode 열
        mode_col = m * ones(N,1);

        % 최종 저장 데이터
        % Mux 순서:
        % [I_meas, HF_energy, HF_rms, hf_norm, arc_flag, arc_detect, T_meas, mode]
        data = [
            I_data, ...
            HF_energy_data, ...
            HF_rms_data, ...
            hf_norm_data, ...
            arc_flag_data, ...
            arc_detect_data, ...
            T_data, ...
            mode_col
        ];

        % CSV 저장
        csvName = fullfile(saveFolder, sprintf("mode%d_run%02d.csv", m, runIdx));
        header = {
            'I_meas', ...
            'HF_energy', ...
            'HF_rms', ...
            'hf_norm', ...
            'arc_flag', ...
            'arc_detect', ...
            'T_meas', ...
            'mode'
        };

        writecell(header, csvName);
        writematrix(data, csvName, "WriteMode", "append");

        fprintf("[SAVED] %s | size = [%d x %d]\n", csvName, size(data,1), size(data,2));
    end
end

disp("모든 CSV 저장이 완료되었습니다.");

%% ===== 로컬 함수 =====
function blk = findModeBlock(modelName, hint)
    blks = find_system(modelName, ...
        "SearchDepth", 3, ...
        "BlockType", "Constant", ...
        "Name", "MODE");
    if ~isempty(blks)
        blk = string(blks{1});
        return;
    end

    blks = find_system(modelName, ...
        "SearchDepth", 3, ...
        "BlockType", "Constant", ...
        "Name", "mode_cmd");
    if ~isempty(blks)
        blk = string(blks{1});
        return;
    end

    cblks = find_system(modelName, ...
        "SearchDepth", 3, ...
        "BlockType", "Constant");

    for i = 1:numel(cblks)
        nm = lower(string(get_param(cblks{i}, "Name")));
        if contains(nm, lower(hint))
            blk = string(cblks{i});
            return;
        end
    end

    error("mode Constant 블록을 찾지 못했습니다. 이름을 확인하세요.");
end

function [I_data, HF_energy_data, HF_rms_data, hf_norm_data, ...
          arc_flag_data, arc_detect_data, T_data, t_data] = readSignals(simOut)

    names = simOut.who;
    ds = [];

    % -------------------------------------------------
    % 0) simOut 안에 dataset이 직접 있는 경우
    % -------------------------------------------------
    if any(strcmp(names, "dataset"))
        ds = simOut.get("dataset");
        fprintf("[INFO] simOut의 dataset 사용\n");
    end

    % -------------------------------------------------
    % 1) simOut 안에 dataset_out이 직접 있는 경우
    % -------------------------------------------------
    if isempty(ds) && any(strcmp(names, "dataset_out"))
        ds = simOut.get("dataset_out");
        fprintf("[INFO] simOut의 dataset_out 사용\n");
    end

    % -------------------------------------------------
    % 2) simOut 안에 out이 있고, out.dataset이 있는 경우
    % -------------------------------------------------
    if isempty(ds) && any(strcmp(names, "out"))
        outVar = simOut.get("out");
        if isstruct(outVar) && isfield(outVar, "dataset")
            ds = outVar.dataset;
            fprintf("[INFO] simOut의 out.dataset 사용 (struct)\n");
        else
            try
                ds = outVar.dataset;
                fprintf("[INFO] simOut의 out.dataset 사용 (object)\n");
            catch
            end
        end
    end

    % -------------------------------------------------
    % 3) base workspace의 out.dataset fallback
    % -------------------------------------------------
    if isempty(ds) && evalin("base", "exist('out','var')")
        outVar = evalin("base", "out");
        if isstruct(outVar) && isfield(outVar, "dataset")
            ds = outVar.dataset;
            fprintf("[INFO] base workspace의 out.dataset 사용 (struct)\n");
        else
            try
                ds = outVar.dataset;
                fprintf("[INFO] base workspace의 out.dataset 사용 (object)\n");
            catch
            end
        end
    end

    if isempty(ds)
        error("simOut 안에도 dataset / dataset_out이 없고, out.dataset도 찾지 못했습니다.");
    end

    fprintf("[DEBUG] class(ds) = %s\n", class(ds));

    % -------------------------------------------------
    % Case 0) double / numeric 형태
    % -------------------------------------------------
    if isnumeric(ds)
        fprintf("[DEBUG] numeric array detected\n");
        vals = squeeze(ds);
        fprintf("[DEBUG] size(ds) = [%s]\n", num2str(size(vals)));

        if isvector(vals)
            error("dataset이 numeric이긴 하지만 벡터입니다. 최소 7개 신호가 필요합니다.");
        end

        % 보통 [N x 7] 또는 [7 x N]
        if size(vals, 2) >= 7
            I_data          = vals(:,1);
            HF_energy_data  = vals(:,2);
            HF_rms_data     = vals(:,3);
            hf_norm_data    = vals(:,4);
            arc_flag_data   = vals(:,5);
            arc_detect_data = vals(:,6);
            T_data          = vals(:,7);
        elseif size(vals, 1) >= 7
            I_data          = vals(1,:).';
            HF_energy_data  = vals(2,:).';
            HF_rms_data     = vals(3,:).';
            hf_norm_data    = vals(4,:).';
            arc_flag_data   = vals(5,:).';
            arc_detect_data = vals(6,:).';
            T_data          = vals(7,:).';
        else
            error("numeric dataset의 차원이 예상과 다릅니다. size(ds) = [%s]", num2str(size(vals)));
        end

        if any(strcmp(names, "tout"))
            t_data = extractTimeData(simOut.get("tout"), I_data);
        else
            t_data = (0:numel(I_data)-1).';
        end
        return;
    end

    % -------------------------------------------------
    % Case 1) Simulink.SimulationData.Dataset 객체
    % -------------------------------------------------
    if isa(ds, "Simulink.SimulationData.Dataset")
        fprintf("[DEBUG] Dataset object detected\n");
        printDatasetNames(ds);

        nElem = getDatasetNumElements(ds);
        if nElem < 7
            error("Dataset element 개수가 부족합니다. 현재 개수: %d, 필요 개수: 7", nElem);
        end

        % Mux 입력 순서 기준
        % 1: I_meas
        % 2: HF_energy
        % 3: HF_rms
        % 4: hf_norm
        % 5: arc_flag
        % 6: arc_detect
        % 7: T_meas
        I_data          = extractSignalData(ds{1}.Values);
        HF_energy_data  = extractSignalData(ds{2}.Values);
        HF_rms_data     = extractSignalData(ds{3}.Values);
        hf_norm_data    = extractSignalData(ds{4}.Values);
        arc_flag_data   = extractSignalData(ds{5}.Values);
        arc_detect_data = extractSignalData(ds{6}.Values);
        T_data          = extractSignalData(ds{7}.Values);

        if any(strcmp(names, "tout"))
            t_data = extractTimeData(simOut.get("tout"), I_data);
        else
            t_data = getTimeFromAny(ds{1}.Values, I_data);
        end
        return;
    end

    % -------------------------------------------------
    % Case 2) struct 형태
    % -------------------------------------------------
    if isstruct(ds)
        fprintf("[DEBUG] struct detected\n");

        % 2-1) ds.signals.values 형태
        if isfield(ds, "signals") && isfield(ds.signals, "values")
            vals = ds.signals.values;
            vals = squeeze(vals);

            fprintf("[DEBUG] size(ds.signals.values) = [%s]\n", num2str(size(vals)));

            if size(vals,2) >= 7
                I_data          = vals(:,1);
                HF_energy_data  = vals(:,2);
                HF_rms_data     = vals(:,3);
                hf_norm_data    = vals(:,4);
                arc_flag_data   = vals(:,5);
                arc_detect_data = vals(:,6);
                T_data          = vals(:,7);
            elseif size(vals,1) >= 7
                I_data          = vals(1,:).';
                HF_energy_data  = vals(2,:).';
                HF_rms_data     = vals(3,:).';
                hf_norm_data    = vals(4,:).';
                arc_flag_data   = vals(5,:).';
                arc_detect_data = vals(6,:).';
                T_data          = vals(7,:).';
            else
                error("ds.signals.values의 차원이 예상과 다릅니다.");
            end

            if isfield(ds, "time")
                t_data = ds.time(:);
            elseif any(strcmp(names, "tout"))
                t_data = extractTimeData(simOut.get("tout"), I_data);
            else
                t_data = (0:numel(I_data)-1).';
            end
            return;
        end

        % 2-2) 구조체 필드별 저장 형태
        fields = fieldnames(ds);
        fprintf("[DEBUG] struct fields: %s\n", strjoin(fields, ", "));

        I_data          = getStructFieldSignal(ds, ["I_meas","I_sig","I_true"]);
        HF_energy_data  = getStructFieldSignal(ds, ["HF_energy","HF_sig"]);
        HF_rms_data     = getStructFieldSignal(ds, ["HF_rms"]);
        hf_norm_data    = getStructFieldSignal(ds, ["hf_norm","HF_norm"]);
        arc_flag_data   = getStructFieldSignal(ds, ["arc_flag"]);
        arc_detect_data = getStructFieldSignal(ds, ["arc_detect"]);
        T_data          = getStructFieldSignal(ds, ["T_meas","T_sig","T_body"]);

        if isfield(ds, "time")
            t_data = ds.time(:);
        elseif any(strcmp(names, "tout"))
            t_data = extractTimeData(simOut.get("tout"), I_data);
        else
            t_data = (0:numel(I_data)-1).';
        end
        return;
    end

    % -------------------------------------------------
    % Case 3) cell 형태
    % -------------------------------------------------
    if iscell(ds)
        fprintf("[DEBUG] cell detected\n");
        if numel(ds) < 7
            error("cell dataset의 원소 개수가 7보다 작습니다.");
        end

        I_data          = extractSignalData(ds{1});
        HF_energy_data  = extractSignalData(ds{2});
        HF_rms_data     = extractSignalData(ds{3});
        hf_norm_data    = extractSignalData(ds{4});
        arc_flag_data   = extractSignalData(ds{5});
        arc_detect_data = extractSignalData(ds{6});
        T_data          = extractSignalData(ds{7});

        if any(strcmp(names, "tout"))
            t_data = extractTimeData(simOut.get("tout"), I_data);
        else
            t_data = (0:numel(I_data)-1).';
        end
        return;
    end

    error("지원하지 않는 dataset 타입입니다: %s", class(ds));
end

function n = getDatasetNumElements(ds)
    try
        n = ds.numElements;
    catch
        try
            n = numElements(ds);
        catch
            try
                n = ds.NumElements;
            catch
                error("Dataset element 개수를 확인할 수 없습니다.");
            end
        end
    end
end

function printDatasetNames(ds)
    fprintf("[DEBUG] Dataset element names:\n");
    try
        n = getDatasetNumElements(ds);
        for i = 1:n
            elem = ds{i};
            try
                fprintf("  %d: %s\n", i, string(elem.Name));
            catch
                fprintf("  %d: (name unavailable)\n", i);
            end
        end
    catch ME
        fprintf("[DEBUG] Dataset 이름 출력 실패: %s\n", ME.message);
    end
end

function x = getStructFieldSignal(s, candNames)
    fns = fieldnames(s);
    for k = 1:numel(candNames)
        name = char(candNames(k));
        idx = find(strcmpi(fns, name), 1);
        if ~isempty(idx)
            x = extractSignalData(s.(fns{idx}));
            fprintf("[INFO] struct에서 '%s' 읽음\n", fns{idx});
            return;
        end
    end
    error("struct에서 후보 신호를 찾지 못했습니다: %s", strjoin(cellstr(candNames), ", "));
end

function t = getTimeFromAny(sig, ref)
    try
        if isa(sig, "timeseries")
            t = sig.Time(:);
            return;
        elseif isstruct(sig) && isfield(sig, "time")
            t = sig.time(:);
            return;
        end
    catch
    end
    t = (0:numel(ref)-1).';
end

function x = extractSignalData(sig)
    if isa(sig, "timeseries")
        x = sig.Data;
    elseif isnumeric(sig)
        x = sig;
    elseif isstruct(sig) && isfield(sig, "signals")
        x = sig.signals.values;
    else
        try
            x = sig.Data;
        catch
            error("신호 데이터를 추출할 수 없습니다. 타입: %s", class(sig));
        end
    end

    x = squeeze(x);
    x = x(:);
end

function t_out = extractTimeData(t, ref)
    if isempty(t)
        t_out = (0:numel(ref)-1).';
        return;
    end

    if isa(t, "timeseries")
        t_out = t.Time;
    elseif isnumeric(t)
        t_out = t(:);
    else
        try
            t_out = t.Time;
        catch
            t_out = (0:numel(ref)-1).';
        end
    end

    t_out = t_out(:);
end
