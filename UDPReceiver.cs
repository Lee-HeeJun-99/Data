using UnityEngine;
using UnityEngine.UI; 
using TMPro;           
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Concurrent;

public class UDPReceiver : MonoBehaviour
{
    Thread receiveThread;
    UdpClient client;
    public int port = 5005;

    [Header("8 ACU Units")]
    public ACUController[] acus = new ACUController[8];

    [Header("Monitor UI (TextMeshPro)")]
    public TextMeshProUGUI statusText; 

    private ConcurrentQueue<string> dataQueue = new ConcurrentQueue<string>();

    void Start()
    {
        InitUDP();
    }

    private void InitUDP()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
        Debug.Log("📡 UDP Receiver Started...");
    }

    private void ReceiveData()
    {
        client = new UdpClient(port);
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, port);

        while (true)
        {
            try
            {
                byte[] data = client.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);
                dataQueue.Enqueue(text); 
            }
            catch (System.Exception) {}
        }
    }

    void Update()
    {
        while (dataQueue.TryDequeue(out string data))
        {
            ParseAndRouteData(data);
        }

        UpdateMonitorUI();
    }

    void UpdateMonitorUI()
    {
        if (statusText == null) return;

        StringBuilder sb = new StringBuilder();
        sb.AppendLine("<color=#FFFF00><b>[ACU MONITORING SYSTEM]</b></color>");
        sb.AppendLine("--------------------------------------------------");

        for (int i = 0; i < acus.Length; i++)
        {
            if (acus[i] != null)
            {
                // 영문 약자로 상태 변환
                string modeStr = GetModeStatus(acus[i].currentMode);
                
                // [ID] M:Mode | T:Temp | A:Arc | S:Stack
                // 크기를 85%로 줄여서 한 줄에 다 들어오게 세팅
                sb.AppendLine($"<size=85%>[{i}] M:{modeStr} | T:{acus[i].temperature:F1}C | A:{acus[i].arcValue:F1} | <color=#FFA500>S:{acus[i].alertStack}</color></size>");
            }
        }

        statusText.text = sb.ToString();
    }

    // 한글 빼고 영문 약자로 리턴
    string GetModeStatus(int mode)
    {
        switch (mode)
        {
            case 0: return "<color=green>OK</color>";
            case 1: return "<color=green>OK</color>";
            case 2: return "<color=yellow>WRN</color>";
            case 3: return "<color=red>DNG</color>";
            default: return "OFF";
        }
    }

    void ParseAndRouteData(string data)
    {
        string[] splitData = data.Split(','); 
        
        if (splitData.Length == 4)
        {
            int id = int.Parse(splitData[0]);
            int mode = int.Parse(splitData[1]);
            float temp = float.Parse(splitData[2]);
            float arc = float.Parse(splitData[3]);

            if (id >= 0 && id < acus.Length)
            {
                if (acus[id] != null)
                {
                    acus[id].UpdateData(mode, temp, arc);
                }
            }
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null) receiveThread.Abort();
        if (client != null) client.Close();
    }
}