#pragma once
#include "Agent.h"

class Seeker : public Agent
{
public:
	__host__ __device__ Seeker(float2 position, float angle, float maxSpeed, float maxForce) : Agent(position, angle), m_maxSpeed(maxSpeed), m_maxForce(maxForce)
	{
	}

	__host__ __device__ Seeker() : Agent{}, m_maxSpeed{}, m_maxForce{} {}

	__host__ __device__ void seek(float2 target)
	{
		float2 destiny = target - m_position;
		destiny = normalize(destiny) * m_maxSpeed;
		applyForce(limit((destiny - m_velocity), m_maxForce));
	}

	__host__ __device__ float getMaxSpeed()const { return m_maxSpeed; };

private:
	float m_maxSpeed;
	float m_maxForce;
};