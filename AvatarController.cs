using UnityEngine;

public class AvatarController : MonoBehaviour
{
    private Animator animator;

    // These variables should match the names of your blendshape or animator parameters.
    public string smileParam = "SmileWeight";
    public string frownParam = "FrownWeight";
    public float transitionSpeed = 3.0f;

    void Start()
    {
        animator = GetComponent<Animator>();
        if (animator == null)
        {
            Debug.LogError("Animator component not found on this GameObject!");
        }
    }

    /// <summary>
    /// Update the avatar's facial expression based on the provided emotion.
    /// This function can be called when you receive a new emotion value (e.g., from a WebSocket).
    /// </summary>
    /// <param name="emotion">Emotion string such as "Happy", "Sad", etc.</param>
    public void UpdateExpression(string emotion)
    {
        switch (emotion)
        {
            case "Happy":
                SetBlendshape(smileParam, 1.0f);
                SetBlendshape(frownParam, 0.0f);
                break;
            case "Sad":
                SetBlendshape(smileParam, 0.0f);
                SetBlendshape(frownParam, 1.0f);
                break;
            case "Angry":
                SetBlendshape(smileParam, 0.0f);
                SetBlendshape(frownParam, 0.8f);
                break;
            case "Surprise":
                SetBlendshape(smileParam, 0.7f);
                SetBlendshape(frownParam, 0.0f);
                break;
            default:
                // Neutral or undefined emotion.
                SetBlendshape(smileParam, 0.5f);
                SetBlendshape(frownParam, 0.5f);
                break;
        }
    }

    private void SetBlendshape(string paramName, float targetValue)
    {
        // Use Lerp for smooth transitions.
        float currentValue = animator.GetFloat(paramName);
        float newValue = Mathf.Lerp(currentValue, targetValue, Time.deltaTime * transitionSpeed);
        animator.SetFloat(paramName, newValue);
    }
}