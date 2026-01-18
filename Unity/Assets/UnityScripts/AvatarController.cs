using UnityEngine;
using UnityEngine.UI;

public class AvatarController : MonoBehaviour
{
    [Header("References")]
    public CVReceiver receiver;
    public Transform headBone; // Head mesh
    public SkinnedMeshRenderer faceMesh; // blendshapes
    public Animator bodyAnimator; // Reference to Animator for body gestures

    [Header("Calibration")]
    // If head rotates wrong way, change to -1
    public float pitchMultiplier = 1.0f; // (X)
    public float yawMultiplier = 1.0f;   // (Y)
    public float rollMultiplier = 1.0f;  // (Z)

    public Vector3 rotationOffset = new Vector3(0, 0, 0);

    [Header("Settings")]
    public float rotationSmooth = 10f;
    public float expressionSmooth = 10f;
    public float movementScale = 1.0f;

    [Header("UI & Audio")]
    public AudioSource audioSource;
    public AudioClip helloClip;
    public AudioClip thankYouClip;
    public AudioClip iLoveYouClip;

    [Header("Blendshape Names")]
    public string shapeSmile = "Smile";
    public string shapeSad = "Sad";
    public string shapeSurprised = "Surprised";
    public string shapeBlink = "Blink"; // Both eyes
    public string shapeAngry = "Angry";
    public string shapeTongue = "TongueOut";
    public string shapeCute = "Cute";

    // Internal vars
    private Quaternion initialRotation;
    private string lastSign = "none";
    private float lastSignTime = 0;
    private string lastGesture = "none";

    // Smoothing vars
    private float currentSmile, currentSad, currentSurprise, currentBlink, currentAngry, currentTongue, currentCute;

    void Start()
    {
        if (headBone != null)
        {
            initialRotation = headBone.localRotation;
        }
    }

    void Update()
    {
        if (receiver == null || receiver.currentData == null) return;
        var data = receiver.currentData;

        // --- 1. Head Rotation ---
        if (data.face_found && headBone != null)
        {
            float p = data.head_pose.pitch;
            float y = data.head_pose.yaw;
            float r = data.head_pose.roll;

            Quaternion targetMovement = Quaternion.Euler(
                p * pitchMultiplier + rotationOffset.x,
                y * yawMultiplier + rotationOffset.y,
                r * rollMultiplier + rotationOffset.z
            );

            Quaternion finalRotation = initialRotation * targetMovement;
            headBone.localRotation = Quaternion.Slerp(headBone.localRotation, finalRotation, Time.deltaTime * rotationSmooth);
        }

        // --- 2. Expressions ---
        if (faceMesh != null)
        {
            float targetSmile = 0, targetSad = 0, targetSurprise = 0;
            float targetBlink = 0, targetAngry = 0, targetTongue = 0; 
            float targetCute = 0;

            float weight = 100f; // Default full weight

            switch (data.expression)
            {
                case "happy": targetSmile = weight; break;
                case "sad": targetSad = weight; break;
                case "surprised": targetSurprise = weight; break;
                case "blink": targetBlink = weight; break;
                case "angry": targetAngry = weight; break;
                case "tongue": targetTongue = weight; break;
                case "cute": targetCute = weight; break;
            }

            // Smoothing (Linear Interpolation)
            currentSmile = Mathf.Lerp(currentSmile, targetSmile, Time.deltaTime * expressionSmooth);
            currentSad = Mathf.Lerp(currentSad, targetSad, Time.deltaTime * expressionSmooth);
            currentSurprise = Mathf.Lerp(currentSurprise, targetSurprise, Time.deltaTime * expressionSmooth);
            currentBlink = Mathf.Lerp(currentBlink, targetBlink, Time.deltaTime * expressionSmooth);
            currentAngry = Mathf.Lerp(currentAngry, targetAngry, Time.deltaTime * expressionSmooth);
            currentTongue = Mathf.Lerp(currentTongue, targetTongue, Time.deltaTime * expressionSmooth);
            currentCute = Mathf.Lerp(currentCute, targetCute, Time.deltaTime * expressionSmooth);

            // Apply
            SetBlendShape(shapeSmile, currentSmile);
            SetBlendShape(shapeSad, currentSad);
            SetBlendShape(shapeSurprised, currentSurprise);
            SetBlendShape(shapeBlink, currentBlink);
            SetBlendShape(shapeAngry, currentAngry);
            SetBlendShape(shapeTongue, currentTongue);
            SetBlendShape(shapeCute, currentCute);
        }

        // --- 3. Body Gestures (Animator) ---
        if (data.gesture != "none" && data.gesture != lastGesture)
        {
            lastGesture = data.gesture;

            // Trigger Animations
            if (bodyAnimator != null)
            {
                switch (data.gesture)
                {
                    case "punch":
                        bodyAnimator.SetTrigger("Punch");
                        break;
                    case "gun":
                        bodyAnimator.SetTrigger("Gun");
                        break;
                    case "wave_horizontal":
                        bodyAnimator.SetTrigger("Wave");
                        break;
                }
            }
        }

        // --- 4. ASL ---
        if (!string.IsNullOrEmpty(data.sign_asl) && data.sign_asl != "none")
        {
            if (data.sign_asl != lastSign || Time.time - lastSignTime > 2.0f)
            {
                lastSign = data.sign_asl;
                lastSignTime = Time.time;

                PlayAudio(data.sign_asl);
            }
        }
    }

    void SetBlendShape(string name, float value)
    {
        int index = faceMesh.sharedMesh.GetBlendShapeIndex(name);
        if (index != -1)
        {
            faceMesh.SetBlendShapeWeight(index, value);
        }
    }

    void PlayAudio(string sign)
    {
        if (audioSource == null) return;

        if (sign == "hello" && helloClip) audioSource.PlayOneShot(helloClip);
        else if (sign == "iloveyou" && iLoveYouClip) audioSource.PlayOneShot(iLoveYouClip);
        else if (sign == "thankyou" && thankYouClip) audioSource.PlayOneShot(thankYouClip);
    }
}
