using UnityEngine;

public class TerminalEffect : MonoBehaviour
{
    public GameObject smokeParticle; // 연기 파티클
    public Renderer wireRenderer;   // 과열될 전선 또는 연결부 렌더러

    private Color normalColor;

    void Awake()
    {
        if (smokeParticle != null) smokeParticle.SetActive(false);
        if (wireRenderer != null) normalColor = wireRenderer.material.color;
    }

    // 🟡 주의 단계: 빨갛게 달아오름
    public void SetWarning(bool isWarning)
    {
        if (wireRenderer == null) return;

        if (isWarning)
        {
            wireRenderer.material.EnableKeyword("_EMISSION");
            // 주황빛으로 은은하게 발광 (달아오르는 느낌)
            wireRenderer.material.SetColor("_EmissionColor", new Color(0.6f, 0.1f, 0f) * 2f);
        }
        else
        {
            wireRenderer.material.DisableKeyword("_EMISSION");
            wireRenderer.material.SetColor("_Color", normalColor);
        }
    }

    // 🔴 위험 단계: 연기 발생
    public void SetDanger(bool isDanger)
    {
        if (smokeParticle != null)
        {
            smokeParticle.SetActive(isDanger);
        }
        
        if (isDanger)
        {
            // 더 강렬하게 붉은색 발광
            wireRenderer.material.SetColor("_EmissionColor", Color.red * 4f);
        }
    }
}