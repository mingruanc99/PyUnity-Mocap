using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System;

[Serializable]
public class CVData
{
    public bool face_found;
    public HeadPose head_pose;
    public string expression;
    public bool hand_found;
    public string gesture;
    public string sign_asl;
    public float sign_conf;
    public string asl_char;
    public string new_asl_char;
    public string current_text;
    // public List<BodyPart> body_pose; // Requires defining BodyPart, leaving out for simplicity unless requested
}

[Serializable]
public class HeadPose
{
    public float pitch;
    public float yaw;
    public float roll;
}

public class CVReceiver : MonoBehaviour
{
    [Header("Network Settings")]
    public int port = 5005;

    [Header("Debug")]
    public bool showDebugLog = true;

    // Private
    private UdpClient client;
    private Thread receiveThread;
    private bool isRunning = true;
    private string lastJson = "";

    // Public access for other scripts
    public CVData currentData;
    [HideInInspector]
    public bool hasNewData = false;

    void Start()
    {
        // Start background thread to avoid freezing Unity
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void ReceiveData()
    {
        try
        {
            client = new UdpClient(port);
            Debug.Log("UDP Receiver started on port " + port);
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to start UDP Client: " + e.Message);
            return;
        }

        while (isRunning)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);

                // Store string to be parsed on main thread
                lastJson = text;
                hasNewData = true;
            }
            catch (Exception e)
            {
                // Socket closed or error
                if (isRunning) Debug.LogWarning(e.ToString());
            }
        }
    }

    void Update()
    {
        if (hasNewData && !string.IsNullOrEmpty(lastJson))
        {
            try
            {
                currentData = JsonUtility.FromJson<CVData>(lastJson);
                hasNewData = false;
                if (showDebugLog)
                {
                    // Optional: Debug.Log($"Expr: {currentData.expression}, Sign: {currentData.sign_asl}");
                }
            }
            catch (Exception e)
            {
                Debug.LogError("JSON Parse Error: " + e.Message);
            }
        }
    }

    void OnDestroy()
    {
        isRunning = false;
        if (client != null) client.Close();
        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Abort();
    }
}
