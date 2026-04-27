
CDoubleMatrix::CDoubleMatrix(int aiRows, int aiColumns)
{
	_Initialize(aiRows, aiColumns, 0, false);
}
CDoubleMatrix::CDoubleMatrix( int aiRows, int aiColumns, double adInitValue )
{
    _Initialize( aiRows, aiColumns, adInitValue, true );
}

void CDoubleMatrix::Assign(double adValue)
{
	for (int iRow = 0; iRow < m_iRows; iRow++)
	{
		for (int iCol = 0; iCol < m_iColumns; iCol++)
		{
			m_ppfData[iRow][iCol] = adValue;
		}
	}
}

// 内存分配与释放
void CDoubleMatrix::_Initialize( int aiRows, int aiColumns, double adInitValue, bool abInit )
{
    if( ( aiRows <= 0 ) || ( aiColumns <= 0 ) )
    {
        throw ( "Invalid number of rows or columns ( Constructor )" );
        m_iRows    = 0;
        m_iColumns = 0;
        return;
    }

    m_iRows    = aiRows;
    m_iColumns = aiColumns;
    m_ppfData = new double*[ m_iRows ];
    for( int i = 0; i < m_iRows; i++ )
    {
        m_ppfData[i] = new double[ m_iColumns ];
    }
    if( ! abInit )
    {
        return;
    }
    Assign( adInitValue );
}


void CDoubleMatrix::Delaunay2D5TriangulationOptimized(CDoubleMatrix& amXw, CDoubleMatrix& amYw, CDoubleMatrix& amZw,
	CDoubleMatrix& amTexture, CDoubleMatrix& amTriangles,
	double dValidThreshold)
{
	int iRows = amXw.Rows();
	int iCols = amXw.Columns();

	// 使用自定义结构体
	size_t maxTriangles = (iRows - 1) * (iCols - 1) * 2;
	std::vector<Triangle> triangles;
	triangles.reserve(maxTriangles);

	for (int i = 0; i < iRows - 1; i++) {
		for (int j = 0; j < iCols - 1; j++) {
			int idx1 = i * iCols + j;
			int idx2 = idx1 + 1;
			int idx3 = idx1 + iCols;
			int idx4 = idx3 + 1;

			bool valid[4] = { (abs(amXw(i, j)) > dValidThreshold &&
				abs(amYw(i, j)) > dValidThreshold &&
				abs(amZw(i, j)) > dValidThreshold),
			(abs(amXw(i, j + 1)) > dValidThreshold &&
				abs(amYw(i, j + 1)) > dValidThreshold &&
				abs(amZw(i, j + 1)) > dValidThreshold),
			(abs(amXw(i + 1, j)) > dValidThreshold &&
				abs(amYw(i + 1, j)) > dValidThreshold &&
				abs(amZw(i + 1, j)) > dValidThreshold),
			(abs(amXw(i + 1, j + 1)) > dValidThreshold &&
				abs(amYw(i + 1, j + 1)) > dValidThreshold &&
				abs(amZw(i + 1, j + 1)) > dValidThreshold) };
			

			int validCount = valid[0] + valid[1] + valid[2] + valid[3];

			if (validCount >= 3) {
				if (validCount == 4) {
					// 直接构造并添加
					triangles.emplace_back(idx1, idx2, idx4);
					triangles.emplace_back(idx1, idx4, idx3);
				}
				else {
					std::vector<int> validIndices;
					validIndices.reserve(4);
					if (valid[0]) validIndices.push_back(idx1);
					if (valid[1]) validIndices.push_back(idx2);
					if (valid[2]) validIndices.push_back(idx3);
					if (valid[3]) validIndices.push_back(idx4);

					if (validIndices.size() == 3) {
						triangles.emplace_back(validIndices[0], validIndices[1], validIndices[2]);
					}
				}
			}
		}
	}

	// 一次性分配结果矩阵
	amTriangles.Resize(triangles.size(), 3);
	for (size_t i = 0; i < triangles.size(); i++) {
		amTriangles(i, 0) = triangles[i].v1;
		amTriangles(i, 1) = triangles[i].v2;
		amTriangles(i, 2) = triangles[i].v3;
	}
}