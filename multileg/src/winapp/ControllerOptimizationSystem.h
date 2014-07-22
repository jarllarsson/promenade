#pragma once
#include "AdvancedEntitySystem.h"
#include "ControllerComponent.h"
#include <ParamChanger.h>

// =======================================================================================
//                                      ControllerOptimizationSystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Measures the controllers and perform parameter evaluation on simulation end.
///			New parameters are provided as well when restarting.
///			Note that fixed frame step size is expected for running optimization sim, 
///			for fully deterministic eval.
///        
/// # ControllerOptimizationSystem
/// 
/// 16-7-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class ControllerOptimizationSystem : public AdvancedEntitySystem
{
private:
	artemis::ComponentMapper<ControllerComponent> controllerComponentMapper;

	ControllerComponent* m_bestScoreController;
	ParamChanger m_changer;
	// Note that fixed frame step size is expected for running optimization sim, for fully deterministic eval
	int m_simTicks; // The amount of ticks the sim will be run
	int m_warmupTicks; // The amount of ticks to ignore taking measurements for score
	int m_currentSimTicks; // The current playback ticks
	bool m_instantEval; // whether to evaluate the score at once (1 sim tick)

	int m_currentBestCandidateIdx;

	double m_lastBestScore; // "Hiscore" (the lower, the better)
	std::vector<double> m_controllerScores; // All scores for one round
	std::vector<float> m_lastBestParams; // saved params needed for a controller to get the hiscore
	std::vector<float> m_paramsMax; // Prefetch of controller parameter
	std::vector<float> m_paramsMin; // bounds in the current sim
	std::vector<std::vector<float> > m_currentParams; // all params for the current controllers

	static int m_testCount; // global amount of executed tests
public:

	ControllerOptimizationSystem();

	virtual void initialize()
	{
		controllerComponentMapper.init(*world);
	};

	virtual void removed(artemis::Entity &e)
	{

	}

	virtual void added(artemis::Entity &e);

	virtual void processEntity(artemis::Entity &e)
	{

	}

	virtual void fixedUpdate(float p_dt)
	{


	}

	static void resetTestCount();
	float getCurrentSimTime();
	bool isSimCompleted();
protected:
private:
	static void incTestCount();

	void restartSim();
	void findCurrentBestCandidate();
	void voidBestCandidate();
	void storeParams();
	void resetScores();
	void perturbParams(int p_offset = 0);
	void evaluateAll();
	double evaluateCandidateFitness(int p_idx);

	bool m_firstControllerAdded;
};

/*
private Controller[] m_optimizableControllers;
    private Controller m_bestScoreController;
    private ParamChanger m_changer;
    public float m_simTime = 1.0f;
    public float m_warmupTime = 1.0f;
    private float m_currentSimTime = 0.0f;
    public bool m_instantEval = false;

    private int m_currentBestCandidate = -1;
    private int m_drawBestCandidate = -1;

    private double m_lastBestScore = double.MaxValue;
    List<float> m_lastBestParams;
    List<List<float>> m_currentParams;

    List<float> m_paramsMax;
    List<float> m_paramsMin;

    private int m_sampleCounter=0;
    public int m_samplesPerIteration = 1;

    private double[] m_totalScores;
    bool m_inited = false;
    bool m_oneRun = false;
    private static bool m_testHandlerCreated = false;
    private static int m_testCount = 0;
    // Use this for initialization
    void Awake()
    {
        if (!m_testHandlerCreated)
        {
            m_currentSimTime = -m_warmupTime;
            m_testHandlerCreated = true;        
            Object.DontDestroyOnLoad(transform.gameObject);
        }
        else if (!m_inited)
        {
            DestroyImmediate(gameObject);
        }
    }

    public float getCurrentSimTime()
    {
        return m_currentSimTime;
    }

    void Init()
    {
        if (m_sampleCounter == 0)
        {
            m_testCount++;
            Debug.Log("Starting new iteration (no." + m_testCount + ")");
        }

        m_sampleCounter++;
        Debug.Log("Sample " + m_sampleCounter);


        
        GameObject[] controllerObjects = GameObject.FindGameObjectsWithTag("optimizable");
        GameObject bestScoreVisualizerObject = GameObject.FindGameObjectWithTag("bestscore");
        m_optimizableControllers = new Controller[controllerObjects.Length];
        for (int i = 0; i < controllerObjects.Length; i++)
        {
            m_optimizableControllers[i] = controllerObjects[i].GetComponent<Controller>();
            //Debug.Log("cobjsC" + m_optimizableControllers[i]);
        }
        m_bestScoreController = bestScoreVisualizerObject.GetComponent<Controller>();
        if (!m_inited)
        {
            m_changer = new ParamChanger();
            m_currentParams = new List<List<float>>();
            m_totalScores = new double[m_optimizableControllers.Length];
            // get bounds for perturbation
            m_paramsMax = m_optimizableControllers[0].GetParamsMax();
            m_paramsMin = m_optimizableControllers[0].GetParamsMin();
            //
            StoreParams();
            m_lastBestParams = new List<float>(m_currentParams[0]);
            PerturbParams(2);
            for (int i = 2; i < m_optimizableControllers.Length; i++)
            {
                IOptimizable opt = m_optimizableControllers[i];
                List<float> paramslist = new List<float>();
                paramslist.AddRange(m_currentParams[i]);
                //Debug.Log("opt "+opt);
               // Debug.Log("params " + m_currentParams[i]);
                opt.ConsumeParams(paramslist); // consume it to controller
            }

            ResetScores();
            m_inited = true;

        }
        else
        {
            for (int i = 0; i < m_optimizableControllers.Length; i++)
            {
                IOptimizable opt = m_optimizableControllers[i];
                //if (opt != null && i<m_currentParams.Count)
                {
                    List<float> paramslist = new List<float>();
                    paramslist.AddRange(m_currentParams[i]);
                    //Debug.Log("current params "+m_currentParams[i]);
                    opt.ConsumeParams(paramslist); // consume it to controller
                }
            }
            if (m_bestScoreController && m_lastBestParams != null && m_lastBestParams.Count>0)
            {
                // and the best score visualizer
                IOptimizable opt = m_bestScoreController;
                List<float> paramslist = new List<float>();
                paramslist.AddRange(m_lastBestParams);
                opt.ConsumeParams(paramslist);
            }
        }
    }


    // Update is called once per frame
    void Update()
    {
        if (!m_oneRun) Init();
        m_currentSimTime += Time.deltaTime;
        if (m_oneRun && (m_instantEval || m_currentSimTime >= m_simTime))
        {            
            EvaluateAll();
            if (m_sampleCounter>=m_samplesPerIteration)
            {
                FindCurrentBestCandidate();
                if (m_currentBestCandidate >= 0)
                    Debug.Log("New best candidate was: " + m_currentBestCandidate + " [" + m_lastBestScore + "p]");
                else
                    Debug.Log("No new candidate (" + m_currentBestCandidate + ")  [" + m_lastBestScore + "p]");
                if (m_lastBestScore > 0.01f)
                    PerturbParams();            
                RestartSim();
            }
            else
            {
                RestartSample();
            }
            // Possible scene restart here <-
        }
        else
            m_oneRun = true;
    }

    private void RestartSim()
    {
        m_oneRun = false;
        m_optimizableControllers = null;
        m_currentSimTime = -m_warmupTime;
        m_sampleCounter = 0;
        //StoreParams();
        VoidBestCandidate();
        ResetScores();
        Application.LoadLevel(0);
    }

    private void RestartSample()
    {
        m_oneRun = false;
        m_optimizableControllers = null;
        m_currentSimTime = -m_warmupTime;
        Application.LoadLevel(0);
    }

    private void FindCurrentBestCandidate()
    {
        VoidBestCandidate();
        double bestScore = m_lastBestScore;
        //Debug.Log("R"+bestScore);
        for (int i = 0; i < m_totalScores.Length; i++)
        {
            //Debug.Log("score("+i+") " + m_totalScores[i]);
            // first make total into mean value
            //m_totalScores[i] /= m_samplesPerIteration;
            if (m_totalScores[i] < bestScore)
            {
                m_currentBestCandidate = i;
                bestScore = m_totalScores[i];
            }
        }
        m_lastBestScore = bestScore;
        if (m_currentBestCandidate > -1)
            m_drawBestCandidate = m_currentBestCandidate;
    }

    private void VoidBestCandidate()
    {
        m_currentBestCandidate = -1;
    }

    private void StoreParams()
    {
        m_currentParams.Clear();
        for (int i = 0; i < m_optimizableControllers.Length; i++)
        {
            m_currentParams.Add(m_optimizableControllers[i].GetParams());
        }
    }

    private void ResetScores()
    {
        for (int i = 0; i < m_totalScores.Length; i++)
        {
            m_totalScores[i] = 0.0f;
        }
    }

    private void PerturbParams(int p_offset=0)
    {
        // Get params from winner and use as basis for perturbing
        // Only update params if they were better than before, else reuse old
        if (m_currentBestCandidate > -1)
        {
            //IOptimizable best = m_optimizableControllers[m_currentBestCandidate];
            m_lastBestParams = m_currentParams[m_currentBestCandidate];
        }
        // Perturb and assign to candidates
        for (int i = p_offset; i < m_optimizableControllers.Length; i++)
        {
            m_currentParams[i] = m_changer.change(m_lastBestParams,m_paramsMin, m_paramsMax, m_testCount); // different perturbation to each
        }                                                           
    }

    private void EvaluateAll()
    {
        
        for (int i = 0; i < m_optimizableControllers.Length; i++)
        {
            //Debug.Log("Eval "+i+" "+m_optimizableControllers[i]);
            m_totalScores[i] += EvaluateCandidateFitness(i);
        }
    }

    private double EvaluateCandidateFitness(int p_idx)
    {
        ControllerMovementRecorder record = m_optimizableControllers[p_idx].m_recordedData;
        double score = record.Evaluate();
        return score;
    }

    public void drawParamGraphs()
    {
        Vector2 s = new Vector2(transform.localScale.x,transform.localScale.y);
        if (m_currentParams!=null)
        {
            
            for (int i = 0; i < m_currentParams.Count; i++)
            {
                
                if (m_currentParams[i]!=null && i<m_optimizableControllers.Length &&
                    m_optimizableControllers[i]!=null)
                {
                    //Debug.Log("p: " + m_currentParams[i].Count);
                    Vector3 pos = m_optimizableControllers[i].transform.position + Vector3.up * 6;
                    drawLineGraph(m_currentParams[i], s, m_optimizableControllers[i].transform.position+Vector3.left*s.x*0.5f);
                }
            }
        }
        if (m_lastBestParams!=null)
            drawLineGraph(m_lastBestParams, s, transform.position);
    }

    public void drawLineGraph(List<float> p_graph, Vector2 p_scale, Vector3 p_wpos)
    {
        for (int n = 0; n < p_graph.Count-1; n++)
        {
            float t = (float)n / (float)p_graph.Count;
            float t1 = (float)(n+1) / (float)p_graph.Count;
            float val = p_graph[n];
            float val1 = p_graph[n+1];
            Gizmos.color=Color.Lerp(Color.cyan,Color.magenta,val*0.01f);
            Gizmos.DrawLine(p_wpos + Vector3.right * p_scale.x * t + Vector3.up * val * p_scale.y,
                p_wpos + Vector3.right * p_scale.x * t1 + Vector3.up * val1 * p_scale.y);
        }
    }

    void OnDrawGizmos()
    {
        if (m_optimizableControllers!=null)
        {
            if (m_drawBestCandidate > -1 && m_drawBestCandidate<m_optimizableControllers.Length &&
                m_optimizableControllers[m_drawBestCandidate] != null)
            {
                if (m_currentBestCandidate > -1)
                {
                    Gizmos.color = Color.green;
                    Gizmos.DrawSphere(m_optimizableControllers[m_drawBestCandidate].transform.position - Vector3.up, 0.3f);
                }
                else
                {
                    Gizmos.color = Color.blue;
                    Gizmos.DrawSphere(m_optimizableControllers[m_drawBestCandidate].transform.position - Vector3.up, 0.3f);
                }
            }
            drawParamGraphs();
        }
    }
*/