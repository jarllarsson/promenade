using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class ControllerMovementRecorder : MonoBehaviour {
    /*
     * 
        The objective function to be minimized is defined as: fobj(P) = wdfd +wvfv +whfh +wrfr
        where fd measures the deviation of the motion from available ref- erence data (m), fv measures 
     * the average deviation from the de- sired speed in both sagittal and coronal directions (m/s), 
     * fh mea- sures head accelerations (m/s2), and fr measures whole body ro- tations (degrees). 
     * These terms are weighted usingwd = 100,wv = 5,wh = 0.5,wr = 5. This objective function is used for 
     * all gaits, although the initial parameter values, initial quadruped state, target velocities, 
     * gait graph, and reference data will be different for each specific gait. 
     * 
     * The fr term measures the
     * difference between the desired heading and the actual heading of the character, as measured in 
     * degrees, by the arccos(x · ˆx), where x and ˆx are the actual and desired forward pointing axes of a leg frame. 
     * This ensures that the quadruped runs forwards rather than sideways. The velocity error term, fv, is 
     * defined as ||v − vd||, and encompasses both sagittal and coronal directions. v is the mean ve- locity as 
     * measured over a stride. The desired velocity in the coronal plane is zero. The desired velocity in the 
     * sagittal plane is an input parameter for the desired gait.

    fobj(P) = +wvfv +(whfh) +wrfr
     * 
     * */

    List<float> m_fvVelocityDeviations; // (current, mean)-desired
    List<float> m_fhHeadAcceleration;
    List<float> m_frBodyRotationDeviations; //arcos(current,desired)
    public float m_fvWeight = 5.0f;
    public float m_fhWeight = 0.5f;
    public float m_frWeight = 5.0f;

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
