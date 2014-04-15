using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class ControllerMovementRecorder : MonoBehaviour 
{
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

    public Controller m_myController;
    List<float> m_fvVelocityDeviations = new List<float>(); // (current, mean)-desired
    List<float> m_fhHeadAcceleration = new List<float>();
    List<float> m_frBodyRotationDeviations = new List<float>(); //arcos(current,desired)
    public float m_fvWeight = 5.0f;
    public float m_fhWeight = 0.5f;
    public float m_frWeight = 5.0f;

    List<Vector3> m_temp_currentStrideVelocities = new List<Vector3>(); // used to calculate mean stride velocity
    List<Vector3> m_temp_currentStrideDesiredVelocities = new List<Vector3>();


	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        fv_calcStrideMeanVelocity();
	}

    void fv_calcStrideMeanVelocity()
    {
        GaitPlayer player = m_myController.m_player;
        bool restarted = player.checkHasRestartedStride_AndResetFlag();
        if (!restarted)
        {
            m_temp_currentStrideVelocities.Add(m_myController.m_currentVelocity);
            m_temp_currentStrideDesiredVelocities.Add(m_myController.m_desiredVelocity);
        }
        else
        {
            Vector3 totalVelocities=Vector3.zero, totalDesiredVelocities=Vector3.zero;
            for (int i = 0; i < m_temp_currentStrideVelocities.Count; i++)
            {
                totalVelocities += m_temp_currentStrideVelocities[i]; 
                // force straight movement behavior from tests, set desired coronal velocity to constant zero:
                totalDesiredVelocities += new Vector3(0.0f,0.0f,m_temp_currentStrideDesiredVelocities[i].z);
            }
            totalVelocities /= (float)m_temp_currentStrideVelocities.Count;
            totalDesiredVelocities /= (float)m_temp_currentStrideDesiredVelocities.Count;
            // add to lists
            m_fvVelocityDeviations.Add(Vector3.Magnitude(totalDesiredVelocities - totalDesiredVelocities));
        }
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
