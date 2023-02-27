#pragma once
#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cerrno>

/// <summary>
/// Shader Class
/// </summary>
class Shader
{
public:
	/// <summary>
	/// Create shader with vertex shader and fragment shader
	/// </summary>
	/// <param name="vertexFile">Vertex shader filename</param>
	/// <param name="fragmentFile">Fragment shader filename</param>
	Shader(const char* vertexFile, const char* fragmentFile);

	/// <summary>
	/// Activate shader
	/// </summary>
	void Activate();

	/// <summary>
	/// Delete shader
	/// </summary>
	void Delete();

public:
	GLuint ID;		//!< Shader ID
};