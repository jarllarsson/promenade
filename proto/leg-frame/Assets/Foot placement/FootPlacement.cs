using UnityEngine;
using System.Collections;

/*  ===================================================================
 *                          Foot placement model
 *  ===================================================================
 *   Feet move from P1 to (P2 = PLF +(v−vd)sfp)
 *   - P1 and P2 are feet start- and end positions
 *   - PLF is the default stepping location relative to the leg frame (saggital and coronal)
 *   - sfp is a scale factor
 *   - v is velocity
 *   - vd is desired velocity
 *   */

public class FootPlacement : MonoBehaviour 
{
    // Tuneable scaling of velocity offset for
    // feet placement
    public float m_tuneVelocityScale = 1.0f;
    public Vector3 m_currentFootPos;


    Vector3 calculateVelocityScaledPos(Vector3 p_footPosLF,
                                       Vector3 p_velocity,
                                       Vector3 p_desiredVelocity)
    {
        return p_footPosLF + (p_velocity - p_desiredVelocity) * m_tuneVelocityScale;
    }
}
