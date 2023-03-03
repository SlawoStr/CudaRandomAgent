#pragma once
#include "Agent.h"

namespace
{
	/// <summary>
	/// Map value from first range to second range
	/// </summary>
	/// <param name="value">Value to map from first range</param>
	/// <param name="start1">First range start</param>
	/// <param name="stop1">First range stop</param>
	/// <param name="start2">Second range start</param>
	/// <param name="stop2">Second range stop</param>
	/// <returns>Mapped value</returns>
	__host__ __device__ float mapValues(float value, float start1, float stop1, float start2, float stop2)
	{
		float newValue = start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
		return newValue;
	}
}

class Arriver : public Agent
{
public:
	__host__ __device__ Arriver(float2 position, float angle, float maxSpeed, float maxForce) : Agent(position, angle), m_maxSpeed(maxSpeed), m_maxForce(maxForce)
	{}

	__host__ __device__ Arriver() : Agent{}, m_maxSpeed{}, m_maxForce{} {}

	__host__ __device__ void arrive(float2 target, float slowingDistance)
	{
		float2 destiny = target - m_position;
		float vecLength = length(destiny);
		destiny = normalize(destiny);
		if (vecLength < slowingDistance)
		{
			float m = mapValues(vecLength, 0, slowingDistance, 0, m_maxSpeed);
			destiny *= m;
		}
		else
		{
			destiny *= m_maxSpeed;
		}
		float2 steer = limit((destiny - m_velocity), m_maxForce);
		applyForce(steer);
	}

	__host__ __device__ float getMaxSpeed()const { return m_maxSpeed; };

private:
	float m_maxSpeed;		//!< Maximum speed
	float m_maxForce;		//!< Maximum turning force
};