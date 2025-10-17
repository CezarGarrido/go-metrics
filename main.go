// go_metrics.go
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Metrics struct {
	File             string
	Entity           string
	EntityType       string // "struct" ou "function"
	LOC              int
	NOM              int
	NOF              int
	WMC              int
	CBO              int
	RFC              int
	LCOM             int
	RefactoringLabel int
}

// countLOC conta linhas de código ignorando comentários e linhas vazias
func countLOC(src string) int {
	lines := strings.Split(src, "\n")
	loc := 0
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l == "" || strings.HasPrefix(l, "//") {
			continue
		}
		loc++
	}
	return loc
}

// collectCalls conta chamadas de função (RFC aproximado)
func collectCalls(fn *ast.FuncDecl) int {
	count := 0
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.CallExpr:
			count++
		}
		return true
	})
	return count
}

// calculateLCOM calcula uma versão simplificada de Lack of Cohesion of Methods
func calculateLCOM(numMethods, numFields int) int {
	if numMethods <= 1 {
		return 0
	}
	if numFields == 0 {
		return numMethods
	}
	return int(math.Max(0, float64(numMethods-numFields)))
}

// analyzeFile analisa um arquivo .go e retorna métricas
func analyzeFile(path string) ([]Metrics, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	loc := countLOC(string(src))

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, path, src, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	var results []Metrics

	// --- structs ---
	ast.Inspect(node, func(n ast.Node) bool {
		ts, ok := n.(*ast.TypeSpec)
		if !ok {
			return true
		}
		st, ok := ts.Type.(*ast.StructType)
		if !ok {
			return true
		}

		// Conta fields
		numFields := 0
		for _, f := range st.Fields.List {
			numFields += len(f.Names)
		}

		numMethods := 0
		wmc := 0
		rfc := 0
		cboSet := make(map[string]struct{})

		// Métodos associados à struct
		for _, decl := range node.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Recv == nil {
				continue
			}
			if len(fd.Recv.List) > 0 {
				if ident, ok := fd.Recv.List[0].Type.(*ast.Ident); ok && ident.Name == ts.Name.Name {
					numMethods++
					wmc++
					rfc += collectCalls(fd)
					// Aprox. de acoplamento (tipos usados nas chamadas)
					ast.Inspect(fd.Body, func(n ast.Node) bool {
						call, ok := n.(*ast.SelectorExpr)
						if ok {
							if id, ok := call.X.(*ast.Ident); ok {
								cboSet[id.Name] = struct{}{}
							}
						}
						return true
					})
				}
			}
		}

		results = append(results, Metrics{
			File:       path,
			Entity:     ts.Name.Name,
			EntityType: "struct",
			LOC:        loc,
			NOM:        numMethods,
			NOF:        numFields,
			WMC:        wmc,
			CBO:        len(cboSet),
			RFC:        rfc,
			LCOM:       calculateLCOM(numMethods, numFields),
		})

		return true
	})

	// --- funções globais ---
	for _, decl := range node.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Recv != nil {
			continue
		}

		results = append(results, Metrics{
			File:       path,
			Entity:     fd.Name.Name,
			EntityType: "function",
			LOC:        loc,
			NOM:        1,
			NOF:        0,
			WMC:        1,
			CBO:        0,
			RFC:        collectCalls(fd),
			LCOM:       0,
		})
	}

	return results, nil
}

// computeRefactoringTarget cria flag binária de necessidade de refatoração
func computeRefactoringTarget(metrics []Metrics) {
	var cboVals, lcomVals, wmcVals, locVals []float64
	for _, m := range metrics {
		cboVals = append(cboVals, float64(m.CBO))
		lcomVals = append(lcomVals, float64(m.LCOM))
		wmcVals = append(wmcVals, float64(m.WMC))
		locVals = append(locVals, float64(m.LOC))
	}

	getQuantile := func(values []float64, q float64) float64 {
		sort.Float64s(values)
		idx := int(q * float64(len(values)))
		if idx >= len(values) {
			idx = len(values) - 1
		}
		return values[idx]
	}

	cboQ := getQuantile(cboVals, 0.75)
	lcomQ := getQuantile(lcomVals, 0.75)
	wmcQ := getQuantile(wmcVals, 0.80)
	locQ := getQuantile(locVals, 0.85)

	for i := range metrics {
		m := &metrics[i]
		if float64(m.CBO) > cboQ ||
			float64(m.LCOM) > lcomQ ||
			float64(m.WMC) > wmcQ ||
			float64(m.LOC) > locQ ||
			(m.NOM > 15 && m.NOF < 3) {
			m.RefactoringLabel = 1
		}
	}
}

// descriptiveStats imprime estatísticas simples no terminal
func descriptiveStats(metrics []Metrics) {
	fields := []string{"LOC", "NOM", "NOF", "WMC", "CBO", "RFC", "LCOM"}
	for _, field := range fields {
		values := []float64{}
		for _, m := range metrics {
			switch field {
			case "LOC":
				values = append(values, float64(m.LOC))
			case "NOM":
				values = append(values, float64(m.NOM))
			case "NOF":
				values = append(values, float64(m.NOF))
			case "WMC":
				values = append(values, float64(m.WMC))
			case "CBO":
				values = append(values, float64(m.CBO))
			case "RFC":
				values = append(values, float64(m.RFC))
			case "LCOM":
				values = append(values, float64(m.LCOM))
			}
		}
		sort.Float64s(values)
		n := len(values)
		if n == 0 {
			continue
		}
		mean := 0.0
		for _, v := range values {
			mean += v
		}
		mean /= float64(n)
		fmt.Printf("%-6s → mean: %.2f | median: %.2f | p75: %.2f | max: %.0f\n",
			field, mean, values[n/2], values[int(float64(n)*0.75)], values[n-1])
	}
}

// saveCSV salva métricas em CSV
func saveCSV(metrics []Metrics, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{"file", "entity", "type", "loc", "nom", "nof", "wmc", "cbo", "rfc", "lcom", "refactoring_needed"}
	_ = w.Write(header)

	for _, m := range metrics {
		row := []string{
			m.File, m.Entity, m.EntityType,
			fmt.Sprint(m.LOC),
			fmt.Sprint(m.NOM),
			fmt.Sprint(m.NOF),
			fmt.Sprint(m.WMC),
			fmt.Sprint(m.CBO),
			fmt.Sprint(m.RFC),
			fmt.Sprint(m.LCOM),
			fmt.Sprint(m.RefactoringLabel),
		}
		_ = w.Write(row)
	}
	return nil
}

func main() {
	flag.Usage = func() {
		fmt.Println("Uso: go run go_metrics.go <diretorio_projetos>")
	}
	flag.Parse()
	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(1)
	}
	root := flag.Arg(0)

	fmt.Println("Extraindo métricas de código Go...")
	var allMetrics []Metrics

	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		if strings.HasSuffix(path, ".go") && !strings.Contains(path, "_test.go") {
			m, err := analyzeFile(path)
			if err == nil {
				allMetrics = append(allMetrics, m...)
			} else {
				fmt.Println("Erro ao processar", path, ":", err)
			}
		}
		return nil
	})

	if len(allMetrics) == 0 {
		fmt.Println("Nenhuma métrica extraída.")
		return
	}

	computeRefactoringTarget(allMetrics)

	fmt.Printf("\nTotal de entidades analisadas: %d\n", len(allMetrics))
	fmt.Println("\nEstatísticas descritivas:")
	descriptiveStats(allMetrics)

	output := "go_metrics.csv"
	if err := saveCSV(allMetrics, output); err == nil {
		fmt.Println("\nDataset salvo em:", output)
	} else {
		fmt.Println("Erro ao salvar CSV:", err)
	}
}
