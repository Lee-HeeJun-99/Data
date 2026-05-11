using UnityEngine;
using TMPro;
using uPLibrary.Networking.M2Mqtt;
using uPLibrary.Networking.M2Mqtt.Messages;
using System.Text;
using System;
using System.Collections.Concurrent;

public class MQTTReceiver : MonoBehaviour
{
    private MqttClient client;
    
    [Header("MQTT Settings")]
    public string brokerIP = "127.0.0.1"; // 브로커 IP (PC면 127.0.0.1)
    public string topic = "acu/data";

    [Header("Connected ACUs")]
    public ACUController[] acus = new ACUController[8];

    [Header("Monitor UI")]
    public TextMeshProUGUI statusText;

    // 데이터 처리를 위한 큐
    private ConcurrentQueue<ACUData> dataQueue = new ConcurrentQueue<ACUData>();

    [Serializable]
    public class ACUData {
        public int id;
        public int mode;
        public float temp;
        public float arc;
    }

    void Start()
    {
        try {
            client = new MqttClient(brokerIP);
            client.MqttMsgPublishReceived += OnMessageReceived;
            
            string clientId = Guid.NewGuid().ToString();
            client.Connect(clientId);
            
            client.Subscribe(new string[] { topic }, new byte[] { MqttMsgBase.QOS_LEVEL_AT_MOST_ONCE });
            Debug.Log("📡 MQTT Connected & Subscribed!");
        } catch (Exception e) {
            Debug.LogError($"MQTT Connection Failed: {e.Message}");
        }
    }

    // MQTT는 별도 스레드에서 작동하므로 큐에 먼저 담습니다.
    void OnMessageReceived(object sender, MqttMsgPublishEventArgs e)
    {
        string message = Encoding.UTF8.GetString(e.Message);
        ACUData receivedData = JsonUtility.FromJson<ACUData>(message);
        dataQueue.Enqueue(receivedData);
    }

    void Update()
    {
        // 1. 큐에서 데이터를 꺼내 실외기 상태 업데이트
        while (dataQueue.TryDequeue(out ACUData data))
        {
            if (data.id >= 0 && data.id < acus.Length && acus[data.id] != null)
            {
                acus[data.id].UpdateData(data.mode, data.temp, data.arc);
            }
        }

        // 2. 모니터 UI 갱신 (아까 만든 영문 약자 버전)
        UpdateMonitorUI();
    }

    void UpdateMonitorUI()
    {
        if (statusText == null) return;
        
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("<color=#FFFF00><b>[ACU MQTT MONITORING]</b></color>");
        sb.AppendLine("--------------------------------------------------");

        for (int i = 0; i < acus.Length; i++)
        {
            if (acus[i] != null)
            {
                string mLabel = GetModeLabel(acus[i].currentMode);
                sb.AppendLine($"<size=85%>[{i}] M:{mLabel} | T:{acus[i].temperature:F1}C | A:{acus[i].arcValue:F1} | S:{acus[i].alertStack}</size>");
            }
        }
        statusText.text = sb.ToString();
    }

    string GetModeLabel(int mode)
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

    void OnApplicationQuit()
    {
        if (client != null && client.IsConnected) client.Disconnect();
    }
}