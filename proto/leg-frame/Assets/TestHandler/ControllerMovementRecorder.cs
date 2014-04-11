using UnityEngine;
using System.Collections;

public class ControllerMovementRecorder : MonoBehaviour {

	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () {
	
	}

    public double Evaluate()
    {
        /*
         * 
        double sumdist = 0.0f;
        double scale = m_resetScale;
        double[] distances = new double[s_size];
        for (int i = 0; i < s_size; i++)
        {
            float t = getTimeFromIdx(i);
            double gy = scale * (double)Mathf.Sin(t * 2.0f * Mathf.PI);
                //((Mathf.Cos(t * 2.0f * Mathf.PI) - 1.0f) * -0.5f);
                //
            double y = m_tuneDataPoints[i];
            double distdiff = Math.Abs(gy - y);
            distances[i] = distdiff;
            sumdist += distdiff;
        }
        double avg = sumdist / (double)s_size;
        double totmeandiffsqr = 0.0f;
        for (int i = 0; i < s_size; i++)
        {
            double mdiff=distances[i]-avg;
            totmeandiffsqr += mdiff * mdiff;
        }
        double sdeviation = Math.Sqrt(totmeandiffsqr / (double)s_size);
        return sumdist*sumdist+10.0f*sdeviation*sdeviation;
         */
        return 1.0;
    }
}
