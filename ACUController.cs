using UnityEngine;

public class ACUController : MonoBehaviour
{
    [Header("내 실외기 번호 (0~7)")]
    public int acuID; 

    [Header("현재 수신 데이터")]
    public int currentMode = 0;
    public float temperature = 25.0f;
    public float arcValue = 0.0f;

    [Header("알림 스택 로직")]
    public int alertStack = 0;
    private float dangerTimer = 0.0f;

    [Header("캡슐 (상태 알림 LED)")]
    public Renderer capsuleRenderer; // 💊 캡슐 자기 자신

    [Header("전기 연결부 (현장 효과)")]
    public Renderer terminalCylinder; // 📍 새로 배치한 실린더
    public GameObject smokeVFX;      // 💨 VFX URP 연기 효과

    void Start()
    {
        if (smokeVFX != null) smokeVFX.SetActive(false); // 시작할 땐 연기 끄기
    }

    void Update()
    {
        CheckDangerTimer();
        UpdateVisuals();
    }

    public void UpdateData(int mode, float temp, float arc)
    {
        currentMode = mode;
        temperature = temp;
        arcValue = arc;
    }

    void CheckDangerTimer()
    {
        if (currentMode == 2 || currentMode == 3)
        {
            dangerTimer += Time.deltaTime;
            if (dangerTimer >= 3.0f) // 3초 지속 시 스택 추가
            {
                alertStack++;
                dangerTimer = 0.0f;
            }
        }
        else
        {
            dangerTimer = 0.0f; 
        }
    }

    void UpdateVisuals()
    {
        if (capsuleRenderer == null || terminalCylinder == null) return;

        // 🟢 0스택: 둘 다 평온
        if (alertStack == 0)
        {
            capsuleRenderer.material.color = Color.green; // 캡슐: 녹색
            terminalCylinder.material.color = Color.white; // 실린더: 기본(흰색/회색)
            terminalCylinder.material.DisableKeyword("_EMISSION");
            if (smokeVFX != null) smokeVFX.SetActive(false);
        }
        // 🟡 1~3스택: 캡슐은 노랑, 실린더는 빨강(과열)
        else if (alertStack >= 1 && alertStack <= 3)
        {
            capsuleRenderer.material.color = Color.yellow; // 캡슐: 노랑 LED
            
            // 🔥 실린더: 빨갛게 달아오름!
            terminalCylinder.material.color = Color.red; 
            terminalCylinder.material.EnableKeyword("_EMISSION");
            terminalCylinder.material.SetColor("_EmissionColor", Color.red * 2.0f); 
            
            if (smokeVFX != null) smokeVFX.SetActive(false);
        }
        // 🔴 4스택 이상: 캡슐은 레드, 실린더는 연기 뿜뿜!
        else if (alertStack >= 4)
        {
            capsuleRenderer.material.color = Color.red; // 캡슐: 레드 LED
            
            // 💨 실린더: 연기 효과 활성화!
            if (smokeVFX != null && !smokeVFX.activeSelf) 
            {
                smokeVFX.SetActive(true);
            }
            
            // 실린더 색상은 계속 빨간색 유지 (또는 더 어둡게 타버린 느낌으로)
            terminalCylinder.material.color = new Color(0.2f, 0f, 0f); 
        }
    }
}